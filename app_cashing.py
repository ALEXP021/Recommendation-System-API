import fastapi
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import List, Optional, Literal, Dict, Any
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import hashlib
import json
from functools import wraps
import os
import sys

# ===== ИСПРАВЛЕННАЯ НАСТРОЙКА ЛОГИРОВАНИЯ =====
# Сначала настраиваем базовое логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Затем создаем директорию для логов
log_dir = './logs'
try:
    os.makedirs(log_dir, exist_ok=True)
    print(f"✅ Директория логов создана: {log_dir}")
    
    # Теперь добавляем файловый handler
    file_handler = logging.FileHandler(f'{log_dir}/api.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
except Exception as e:
    print(f"⚠️ Не удалось создать директорию логов: {e}")

# Модели для запросов
class UserRecommendationRequest(BaseModel):
    visitorid: int = Field(..., description="ID пользователя")
    top_k: Optional[int] = Field(None, description="Количество рекомендаций")
    min_confidence: Optional[float] = Field(None, description="Минимальная уверенность (0-1)")
    confidence_level: Optional[Literal["high", "medium", "low", "all"]] = Field(None, description="Уровень уверенности")

class ItemRecommendationRequest(BaseModel):
    itemid: int = Field(..., description="ID товара")
    top_k: Optional[int] = Field(None, description="Количество пользователей")
    min_score: Optional[float] = Field(None, description="Минимальный score")
    confidence_level: Optional[Literal["high", "medium", "low", "all"]] = Field(None, description="Уровень уверенности")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус API")
    model_type: str = Field(..., description="Тип модели")
    model_loaded: bool = Field(..., description="Модель загружена")
    total_items: int = Field(..., description="Всего товаров")
    timestamp: str = Field(..., description="Время запроса")
    environment: str = Field(..., description="Окружение")

class CacheStatsResponse(BaseModel):
    hit_rate: float = Field(..., description="Процент попаданий в кеш")
    hits: int = Field(..., description="Количество попаданий")
    misses: int = Field(..., description="Количество промахов")
    total_requests: int = Field(..., description="Всего запросов")
    cache_size: int = Field(..., description="Размер кеша")

# УЛУЧШЕННЫЙ IN-MEMORY КЕШ
class MemoryCache:
    def __init__(self, max_size: int = 5000, cleanup_interval: int = 100):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.operations_count = 0
        logger.info(f"💾 Инициализирован In-Memory кеш (максимум {max_size} записей)")
    
    def get(self, key: str) -> Any:
        """Получить значение из кеша"""
        try:
            if key in self._cache:
                data, expiry = self._cache[key]
                if datetime.now() < expiry:
                    self.hits += 1
                    return data
                else:
                    # Автоматически удаляем просроченные записи
                    del self._cache[key]
                    logger.debug(f"🗑️ Удалена просроченная запись: {key}")
            self.misses += 1
            return None
        except Exception as e:
            logger.error(f"❌ Ошибка получения из кеша: {e}")
            return None
    
    def set(self, key: str, data: Any, ttl: int = 3600):
        """Установить значение в кеш"""
        try:
            expiry = datetime.now() + timedelta(seconds=ttl)
            self._cache[key] = (data, expiry)
            
            # Периодическая очистка при достижении лимита
            self.operations_count += 1
            if len(self._cache) > self.max_size or self.operations_count % self.cleanup_interval == 0:
                self._cleanup()
                
        except Exception as e:
            logger.error(f"❌ Ошибка установки в кеш: {e}")
    
    def delete_pattern(self, pattern: str) -> int:
        """Удалить ключи по паттерну"""
        try:
            keys_to_delete = [key for key in self._cache.keys() if pattern in key]
            for key in keys_to_delete:
                del self._cache[key]
            logger.info(f"🗑️ Удалено {len(keys_to_delete)} записей по паттерну: {pattern}")
            return len(keys_to_delete)
        except Exception as e:
            logger.error(f"❌ Ошибка удаления по паттерну: {e}")
            return 0
    
    def _cleanup(self):
        """Очистка устаревших записей"""
        try:
            now = datetime.now()
            expired_keys = [key for key, (_, expiry) in self._cache.items() if now >= expiry]
            for key in expired_keys:
                del self._cache[key]
                
            # Если все еще превышен лимит, удаляем самые старые записи
            if len(self._cache) > self.max_size:
                keys_to_remove = list(self._cache.keys())[:len(self._cache) - self.max_size]
                for key in keys_to_remove:
                    del self._cache[key]
                logger.info(f"🧹 Очистка кеша: удалено {len(expired_keys) + len(keys_to_remove)} записей")
            elif expired_keys:
                logger.debug(f"🧹 Очищено {len(expired_keys)} устаревших записей кеша")
                
        except Exception as e:
            logger.error(f"❌ Ошибка очистки кеша: {e}")
    
    def hit_rate(self) -> float:
        """Рассчитать процент попаданий"""
        total = self.hits + self.misses
        return round(self.hits / total * 100, 2) if total > 0 else 0.0
    
    def size(self) -> int:
        """Текущий размер кеша"""
        return len(self._cache)
    
    def clear(self):
        """Полная очистка кеша"""
        self._cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("🧹 Кеш полностью очищен")

# Глобальные переменные
MODEL = None
SCALER = None
FEATURE_NAMES = None
TOTAL_ITEMS = 466868
ITEMS_GLOBAL_RATING = None
MEMORY_CACHE = MemoryCache(max_size=5000)

# ДЕКОРАТОР ДЛЯ КЕШИРОВАНИЯ
def cache_response(ttl: int = 3600, prefix: str = ""):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Создаем ключ кеша на основе аргументов
                cache_key_data = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{prefix}:{func.__name__}:{hashlib.md5(cache_key_data.encode()).hexdigest()}"
                
                # Пробуем получить из кеша
                cached = MEMORY_CACHE.get(cache_key)
                if cached is not None:
                    logger.debug(f"📦 Cache HIT for {func.__name__}")
                    return cached
                
                # Выполняем функцию и кешируем результат
                result = await func(*args, **kwargs)
                MEMORY_CACHE.set(cache_key, result, ttl)
                logger.debug(f"💾 Cache SET for {func.__name__} (TTL: {ttl}s)")
                
                return result
            except Exception as e:
                logger.error(f"❌ Ошибка кеширования в {func.__name__}: {e}")
                # В случае ошибки кеширования просто выполняем функцию
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# Функции инвалидации кеша
def invalidate_user_cache(visitorid: int):
    """Инвалидировать кеш для пользователя"""
    pattern = f"user:recommend_user_to_items:*{visitorid}*"
    deleted_count = MEMORY_CACHE.delete_pattern(pattern)
    logger.info(f"🗑️ Инвалидирован кеш для пользователя {visitorid}: {deleted_count} ключей")
    return deleted_count

def invalidate_item_cache(itemid: int):
    """Инвалидировать кеш для товара"""
    patterns = [
        f"item:recommend_item_to_users:*{itemid}*",
        f"score:get_item_score:*{itemid}*"
    ]
    total_deleted = 0
    for pattern in patterns:
        total_deleted += MEMORY_CACHE.delete_pattern(pattern)
    logger.info(f"🗑️ Инвалидирован кеш для товара {itemid}: {total_deleted} ключей")
    return total_deleted

def invalidate_global_rating_cache():
    """Инвалидировать кеш глобального рейтинга"""
    pattern = "global:get_global_rating:"
    deleted_count = MEMORY_CACHE.delete_pattern(pattern)
    logger.info(f"🗑️ Инвалидирован кеш глобального рейтинга: {deleted_count} ключей")
    return deleted_count

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Улучшенная функция инициализации приложения"""
    global MODEL, SCALER, FEATURE_NAMES, ITEMS_GLOBAL_RATING
    
    logger.info("🚀 Запуск инициализации приложения...")
    
    try:
        # Определяем путь к модели
        model_path = os.getenv('MODEL_PATH', 'best_model_fm.pkl')
        logger.info(f"🔍 Поиск модели по пути: {model_path}")
        
        # Проверяем существование файла модели
        if not os.path.exists(model_path):
            logger.warning(f"⚠️ Файл модели не найден: {model_path}")
            logger.info("📝 Будет использован демо-режим")
            MODEL = None
            SCALER = None
        else:
            # Загрузка модели
            logger.info("📦 Загрузка модели...")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            MODEL = model_data['model']
            SCALER = model_data['scaler']
            logger.info("✅ Модель успешно загружена")
        
        # Инициализация фичей
        FEATURE_NAMES = [
            'user_item_click_count', 'user_item_cart_count', 'user_item_purchase_count',
            'days_since_last_user_item_interaction', 'user_item_recency',
            'user_item_interaction_score_normalized', 'user_item_similarity_normalized',
            'user_item_property_similarity_normalized', 'item_popularity_global',
            'item_conversion_rate', 'item_popularity_trend', 'cart_to_view_ratio',
            'item_unique_users', 'item_popularity_7d', 'item_popularity_30d',
            'item_age_days', 'is_new_item', 'efficiency_score',
            'user_cat_affinity_vs_item_cat', 'item_in_user_top_categories',
            'user_activity_time_match', 'user_activity_trend_7d',
            'embedding_mean', 'embedding_std', 'embedding_max', 'embedding_min', 'embedding_norm'
        ]
        
        # Инициализация глобального рейтинга
        logger.info("📊 Инициализация глобального рейтинга...")
        ITEMS_GLOBAL_RATING = initialize_global_rating()
        
        logger.info("🎯 Инициализация завершена успешно!")
        logger.info(f"✅ Модель загружена: {MODEL is not None}")
        logger.info(f"📊 Товаров в системе: {TOTAL_ITEMS}")
        logger.info(f"💾 In-Memory кеш готов (размер: {MEMORY_CACHE.size()})")
        logger.info(f"🌍 Окружение: {os.getenv('ENVIRONMENT', 'development')}")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка инициализации: {e}")
        MODEL = None
        SCALER = None

    yield
    
    logger.info("👋 Завершение работы приложения...")

def initialize_global_rating():
    """Инициализация глобального рейтинга товаров с улучшенной обработкой ошибок"""
    try:
        logger.info("🎲 Генерация глобального рейтинга...")
        np.random.seed(42)
        global_rating = {}
        
        # Для производительности генерируем только часть товаров
        sample_size = min(10000, TOTAL_ITEMS)  # Ограничиваем для скорости
        item_ids = np.random.choice(range(1, TOTAL_ITEMS + 1), sample_size, replace=False)
        
        for item_id in item_ids:
            base_rating = max(0.1, 1.0 - (item_id / TOTAL_ITEMS) * 0.8)
            random_component = np.random.normal(0, 0.1)
            final_rating = max(0.01, min(0.99, base_rating + random_component))
            
            global_rating[item_id] = {
                'score': float(final_rating),  # Конвертируем в Python float
                'popularity_rank': 0
            }
        
        # Сортируем и назначаем ранги
        sorted_items = sorted(global_rating.items(), key=lambda x: x[1]['score'], reverse=True)
        for rank, (item_id, _) in enumerate(sorted_items):
            global_rating[item_id]['popularity_rank'] = rank + 1
        
        logger.info(f"📈 Глобальный рейтинг инициализирован для {len(global_rating)} товаров")
        return global_rating
        
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации глобального рейтинга: {e}")
        return {}

def get_confidence_level(score: float) -> str:
    """Определить уровень уверенности по score"""
    if score > 0.7:
        return "high"
    elif score > 0.3:
        return "medium"
    else:
        return "low"

def prepare_features_for_user_item(visitorid: int, item_id: int) -> pd.DataFrame:
    """Подготовка признаков для пары пользователь-товар"""
    try:
        # Базовые признаки пользователя
        np.random.seed(visitorid % 2000000)  # Увеличили для больших ID
        user_base_features = {
            'user_item_click_count': max(1, int(np.random.normal(5, 3))),
            'user_item_cart_count': max(0, int(np.random.normal(2, 1))),
            'user_item_purchase_count': max(0, int(np.random.normal(1, 0.5))),
            'days_since_last_user_item_interaction': np.random.randint(0, 30),
            'user_item_recency': float(np.random.uniform(0, 1)),
            'user_item_interaction_score_normalized': float(np.random.uniform(0, 1)),
            'user_item_similarity_normalized': float(np.random.uniform(0, 1)),
            'user_item_property_similarity_normalized': float(np.random.uniform(0, 1)),
            'user_cat_affinity_vs_item_cat': float(np.random.uniform(0, 1)),
            'item_in_user_top_categories': np.random.randint(0, 2),
            'user_activity_time_match': float(np.random.uniform(0, 1)),
            'user_activity_trend_7d': float(np.random.uniform(-0.5, 0.5)),
        }
        
        # Признаки товара
        np.random.seed(item_id)
        global_score = ITEMS_GLOBAL_RATING.get(item_id, {}).get('score', 0.5) if ITEMS_GLOBAL_RATING else 0.5
        
        item_features = {
            'item_popularity_global': float(global_score * 100),
            'item_conversion_rate': float(global_score * np.random.uniform(0.5, 1.5)),
            'item_popularity_trend': float(np.random.uniform(-0.2, 0.2)),
            'cart_to_view_ratio': float(global_score * np.random.uniform(0.1, 0.3)),
            'item_unique_users': int(global_score * 1000),
            'item_popularity_7d': float(global_score * 100 * np.random.uniform(0.8, 1.2)),
            'item_popularity_30d': float(global_score * 100 * np.random.uniform(0.9, 1.1)),
            'item_age_days': np.random.randint(1, 1000),
            'is_new_item': 1 if np.random.random() > 0.8 else 0,
            'efficiency_score': float(global_score * np.random.uniform(0.7, 1.3)),
        }
        
        # Эмбеддинги
        embedding_features = {
            'embedding_mean': float(global_score - 0.5),
            'embedding_std': float(np.random.uniform(0.1, 0.3)),
            'embedding_max': float(global_score + np.random.uniform(0, 0.2)),
            'embedding_min': float(global_score - np.random.uniform(0, 0.2)),
            'embedding_norm': float(np.random.uniform(0.5, 1.5)),
        }
        
        # Объединяем все признаки
        feature_row = {}
        feature_row.update(user_base_features)
        feature_row.update(item_features)
        feature_row.update(embedding_features)
        
        np.random.seed()
        return pd.DataFrame([feature_row])[FEATURE_NAMES]
    
    except Exception as e:
        logger.error(f"❌ Ошибка подготовки фичей: {e}")
        # Возвращаем демо-фичи при ошибке
        return pd.DataFrame([{name: 0.5 for name in FEATURE_NAMES}])

def predict_score_for_user_item(visitorid: int, item_id: int) -> float:
    """Предсказать score для пары пользователь-товар"""
    try:
        if MODEL is None or SCALER is None:
            # Демо-режим если модель не загружена
            base_score = 0.5
            user_factor = (visitorid % 2000000) / 2000000.0  # Увеличили диапазон
            item_factor = (item_id % 100) / 100.0
            return float((base_score + user_factor * 0.3 + item_factor * 0.2) / 1.5)
        
        features_df = prepare_features_for_user_item(visitorid, item_id)
        features_scaled = SCALER.transform(features_df)
        score = MODEL.predict_proba(features_scaled)[:, 1][0]
        return float(score)  # Конвертируем в Python float
    except Exception as e:
        logger.warning(f"⚠️ Ошибка предсказания для user {visitorid}, item {item_id}: {e}")
        # Возвращаем демо-значение при ошибке
        return float(max(0.1, min(0.9, (visitorid + item_id) % 2000000 / 2000000.0)))

# Определяем окружение
environment = os.getenv('ENVIRONMENT', 'development')

app = FastAPI(
    title="Recommendation System API",
    description="MVP системы рекомендаций с полным функционалом и in-memory кешированием",
    version="1.2.0",
    lifespan=lifespan,
    docs_url="/docs" if environment != "production" else None,
    redoc_url="/redoc" if environment != "production" else None
)

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API endpoints
@app.get("/", summary="Главная страница", description="Информация о API и доступных эндпоинтах")
async def root():
    return {
        "message": "🎉 Recommendation System API с полным функционалом и in-memory кешированием!",
        "status": "active",
        "environment": environment,
        "model_loaded": MODEL is not None,
        "cache_enabled": True,
        "total_items": TOTAL_ITEMS,
        "confidence_levels": {
            "high": "> 0.7",
            "medium": "0.3 - 0.7", 
            "low": "< 0.3"
        },
        "endpoints": {
            "health": "/health",
            "cache_stats": "/cache/stats",
            "cache_clear": "/cache/clear",
            "invalidate_cache": "/cache/invalidate/{type}/{id}",
            "global_rating": "/rating/global?top_k=100&min_score=0.0",
            "user_to_items": "/recommend/user-to-items?visitorid=123&top_k=10",
            "item_to_users": "/recommend/item-to-users?itemid=456&top_k=10", 
            "item_score": "/score/{visitorid}/{item_id}",
            "docs": "/docs"
        },
        "examples": {
            "user_recommendations": "http://localhost:8001/recommend/user-to-items?visitorid=1016904&top_k=10&min_confidence=0.5",
            "item_recommendations": "http://localhost:8001/recommend/item-to-users?itemid=456&top_k=5&confidence_level=high",
            "global_rating": "http://localhost:8001/rating/global?top_k=50&min_score=0.7"
        }
    }

@app.get("/health", response_model=HealthResponse, summary="Проверка здоровья", description="Проверить статус API и модели")
async def health_check():
    return HealthResponse(
        status="healthy" if MODEL else "degraded",
        model_type="Factorization Machines" if MODEL else "Demo Mode",
        model_loaded=MODEL is not None,
        total_items=TOTAL_ITEMS,
        timestamp=datetime.now().isoformat(),
        environment=environment
    )

@app.get("/cache/stats", response_model=CacheStatsResponse, summary="Статистика кеша", description="Получить метрики кеширования")
async def get_cache_stats():
    return CacheStatsResponse(
        hit_rate=MEMORY_CACHE.hit_rate(),
        hits=MEMORY_CACHE.hits,
        misses=MEMORY_CACHE.misses,
        total_requests=MEMORY_CACHE.hits + MEMORY_CACHE.misses,
        cache_size=MEMORY_CACHE.size()
    )

@app.get("/cache/clear", summary="Очистка кеша", description="Полностью очистить кеш")
async def clear_cache():
    """Полная очистка кеша"""
    try:
        MEMORY_CACHE.clear()
        return {"message": "Кеш полностью очищен", "cache_size": 0}
    except Exception as e:
        logger.error(f"❌ Ошибка очистки кеша: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка очистки кеша: {str(e)}")

@app.post("/cache/invalidate/{type}/{id}", summary="Инвалидация кеша", description="Принудительно очистить кеш для пользователя или товара")
async def invalidate_cache(type: Literal["user", "item", "global"], id: int = None):
    try:
        if type == "user":
            deleted_count = invalidate_user_cache(id)
            return {"message": f"Кеш для пользователя {id} инвалидирован", "deleted_keys": deleted_count}
        elif type == "item":
            deleted_count = invalidate_item_cache(id)
            return {"message": f"Кеш для товара {id} инвалидирован", "deleted_keys": deleted_count}
        elif type == "global":
            deleted_count = invalidate_global_rating_cache()
            return {"message": "Кеш глобального рейтинга инвалидирован", "deleted_keys": deleted_count}
        else:
            raise HTTPException(status_code=400, detail="Неверный тип кеша")
    except Exception as e:
        logger.error(f"❌ Ошибка инвалидации кеша: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка инвалидации кеша: {str(e)}")

@app.get("/rating/global", summary="Глобальный рейтинг товаров", description="Получить глобальный рейтинг всех товаров")
@cache_response(ttl=86400, prefix="global")  # 24 часа - глобальный рейтинг меняется редко
async def get_global_rating(
    top_k: int = Query(100, description="Количество товаров в рейтинге", ge=1, le=10000),
    min_score: float = Query(0.0, description="Минимальный score", ge=0.0, le=1.0),
    confidence_level: Optional[Literal["high", "medium", "low", "all"]] = Query("all", description="Уровень уверенности")
):
    """Получить глобальный рейтинг товаров по всей базе"""
    
    if ITEMS_GLOBAL_RATING is None:
        raise HTTPException(status_code=503, detail="Global rating not loaded")
    
    try:
        filtered_items = []
        for item_id, rating_data in ITEMS_GLOBAL_RATING.items():
            score = rating_data['score']
            confidence = get_confidence_level(score)
            
            if score < min_score:
                continue
            if confidence_level != "all" and confidence != confidence_level:
                continue
                
            filtered_items.append((item_id, score, confidence, rating_data['popularity_rank']))
        
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        top_items = filtered_items[:top_k]
        
        rating = [
            {
                "item_id": int(item_id),  # Конвертируем в Python int
                "name": f"Товар {item_id}",
                "global_score": float(score),
                "confidence": confidence,
                "global_rank": int(global_rank),  # Конвертируем в Python int
                "top_k_rank": int(rank + 1)  # Конвертируем в Python int
            }
            for rank, (item_id, score, confidence, global_rank) in enumerate(top_items)
        ]
        
        confidence_stats = {
            "high": len([item for item in top_items if item[2] == "high"]),
            "medium": len([item for item in top_items if item[2] == "medium"]),
            "low": len([item for item in top_items if item[2] == "low"])
        }
        
        logger.info(f"📊 Возвращен глобальный рейтинг {len(rating)} товаров")
        
        return {
            "type": "global_rating",
            "total_items": TOTAL_ITEMS,
            "items_in_rating": len(rating),
            "filters_applied": {
                "top_k": top_k,
                "min_score": min_score,
                "confidence_level": confidence_level
            },
            "statistics": {
                "confidence_distribution": confidence_stats,
                "score_range": {
                    "min": float(min([score for _, score, _, _ in top_items])) if top_items else 0,
                    "max": float(max([score for _, score, _, _ in top_items])) if top_items else 0,
                    "avg": float(np.mean([score for _, score, _, _ in top_items])) if top_items else 0
                }
            },
            "rating": rating,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения глобального рейтинга: {e}")
        raise HTTPException(status_code=500, detail=f"Global rating failed: {str(e)}")

@app.get("/recommend/user-to-items", summary="Товары для пользователя", description="Получить персонализированные рекомендации товаров для пользователя")
@cache_response(ttl=1800, prefix="user")  # 30 минут для пользовательских рекомендаций
async def recommend_user_to_items(
    visitorid: int = Query(..., description="ID пользователя", ge=1, le=2000000),  # Увеличили лимит до 2,000,000
    top_k: int = Query(100, description="Количество рекомендаций", ge=1, le=1000),
    min_confidence: Optional[float] = Query(None, description="Минимальная уверенность (0-1)", ge=0.0, le=1.0),
    confidence_level: Optional[Literal["high", "medium", "low", "all"]] = Query("all", description="Уровень уверенности")
):
    """Рекомендация топ товаров для конкретного пользователя"""
    
    try:
        # Определяем диапазон товаров
        items_to_process = min(top_k * 5, 1000)  # Ограничиваем для производительности
        
        # Выбираем товары для обработки (стратегическая выборка)
        item_ids = list(range(1, items_to_process + 1))
        
        logger.info(f"🔍 Поиск товаров для пользователя {visitorid}")
        
        # Предсказываем scores для всех товаров
        item_scores = []
        for item_id in item_ids:
            score = predict_score_for_user_item(visitorid, item_id)
            item_scores.append((item_id, score))
        
        # СОРТИРУЕМ по score (от высокого к низкому)
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Применяем фильтры
        filtered_items = []
        for item_id, score in item_scores:
            confidence = get_confidence_level(score)
            
            if min_confidence is not None and score < min_confidence:
                continue
            if confidence_level != "all" and confidence != confidence_level:
                continue
                
            filtered_items.append((item_id, score, confidence))
        
        # Берем топ-K
        selected_items = filtered_items[:top_k]
        
        # Формируем ответ
        recommendations = [
            {
                "item_id": int(item_id),  # Конвертируем в Python int
                "name": f"Товар {item_id}",
                "score": float(score),
                "confidence": confidence,
                "rank": int(rank + 1)  # Конвертируем в Python int
            }
            for rank, (item_id, score, confidence) in enumerate(selected_items)
        ]
        
        confidence_stats = {
            "high": len([item for item in selected_items if item[2] == "high"]),
            "medium": len([item for item in selected_items if item[2] == "medium"]),
            "low": len([item for item in selected_items if item[2] == "low"])
        }
        
        logger.info(f"✅ Найдено {len(recommendations)} товаров для пользователя {visitorid}")
        
        return {
            "visitorid": visitorid,
            "recommendations": recommendations,
            "total_items_processed": items_to_process,
            "items_after_filtering": len(filtered_items),
            "items_returned": len(recommendations),
            "filters_applied": {
                "min_confidence": min_confidence,
                "confidence_level": confidence_level
            },
            "statistics": {
                "confidence_distribution": confidence_stats
            },
            "score_range": {
                "min": float(min([score for _, score, _ in selected_items])) if selected_items else 0,
                "max": float(max([score for _, score, _ in selected_items])) if selected_items else 0,
                "avg": float(np.mean([score for _, score, _ in selected_items])) if selected_items else 0
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка рекомендации товаров: {e}")
        raise HTTPException(status_code=500, detail=f"Item recommendation failed: {str(e)}")

@app.get("/recommend/item-to-users", summary="Пользователи для товара", description="Найти пользователей для рекомендации конкретного товара")
@cache_response(ttl=3600, prefix="item")  # 1 час для товарных рекомендаций
async def recommend_item_to_users(
    itemid: int = Query(..., description="ID товара", ge=1, le=466868),
    top_k: int = Query(100, description="Количество пользователей", ge=1, le=1000),
    min_score: Optional[float] = Query(None, description="Минимальный score", ge=0.0, le=1.0),
    confidence_level: Optional[Literal["high", "medium", "low", "all"]] = Query("all", description="Уровень уверенности")
):
    """Рекомендация топ пользователей для конкретного товара"""
    
    try:
        # Определяем диапазон пользователей
        users_to_process = min(top_k * 5, 500)  # Ограничиваем для производительности
        
        # Выбираем пользователей для обработки
        user_ids = list(range(1, users_to_process + 1))
        
        logger.info(f"🔍 Поиск пользователей для товара {itemid}")
        
        # Предсказываем scores для всех пользователей
        user_scores = []
        for user_id in user_ids:
            score = predict_score_for_user_item(user_id, itemid)
            user_scores.append((user_id, score))
        
        # СОРТИРУЕМ по score (от высокого к низкому)
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Применяем фильтры
        filtered_users = []
        for user_id, score in user_scores:
            confidence = get_confidence_level(score)
            
            if min_score is not None and score < min_score:
                continue
            if confidence_level != "all" and confidence != confidence_level:
                continue
                
            filtered_users.append((user_id, score, confidence))
        
        # Берем топ-K
        selected_users = filtered_users[:top_k]
        
        # Формируем ответ
        recommendations = [
            {
                "visitorid": int(user_id),  # Конвертируем в Python int
                "name": f"Пользователь {user_id}",
                "score": float(score),
                "confidence": confidence,
                "rank": int(rank + 1)  # Конвертируем в Python int
            }
            for rank, (user_id, score, confidence) in enumerate(selected_users)
        ]
        
        confidence_stats = {
            "high": len([user for user in selected_users if user[2] == "high"]),
            "medium": len([user for user in selected_users if user[2] == "medium"]),
            "low": len([user for user in selected_users if user[2] == "low"])
        }
        
        logger.info(f"✅ Найдено {len(recommendations)} пользователей для товара {itemid}")
        
        return {
            "itemid": itemid,
            "item_name": f"Товар {itemid}",
            "recommended_users": recommendations,
            "total_users_processed": users_to_process,
            "users_after_filtering": len(filtered_users),
            "users_returned": len(recommendations),
            "filters_applied": {
                "min_score": min_score,
                "confidence_level": confidence_level
            },
            "statistics": {
                "confidence_distribution": confidence_stats
            },
            "score_range": {
                "min": float(min([score for _, score, _ in selected_users])) if selected_users else 0,
                "max": float(max([score for _, score, _ in selected_users])) if selected_users else 0,
                "avg": float(np.mean([score for _, score, _ in selected_users])) if selected_users else 0
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка рекомендации пользователей: {e}")
        raise HTTPException(status_code=500, detail=f"User recommendation failed: {str(e)}")

@app.get("/score/{visitorid}/{item_id}", summary="Score товара", description="Получить score конкретного товара для пользователя")
@cache_response(ttl=7200, prefix="score")  # 2 часа для пар user-item
async def get_item_score(visitorid: int, item_id: int):
    """Получить score одного товара для пользователя"""
    
    if item_id < 1 or item_id > TOTAL_ITEMS:
        raise HTTPException(status_code=404, detail=f"Item not found. Available: 1-{TOTAL_ITEMS}")
    
    try:
        score = predict_score_for_user_item(visitorid, item_id)
        
        global_rating = ITEMS_GLOBAL_RATING.get(item_id, {}) if ITEMS_GLOBAL_RATING else {}
        global_score = global_rating.get('score', 0)
        global_rank = global_rating.get('popularity_rank', 0)
        
        return {
            "visitorid": visitorid,
            "item_id": item_id,
            "personal_score": float(score),
            "global_score": float(global_score),
            "global_rank": int(global_rank),  # Конвертируем в Python int
            "confidence": get_confidence_level(score),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения score: {e}")
        raise HTTPException(status_code=500, detail=f"Score calculation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8001))  # Изменен порт на 8001
    
    print("🚀 Запуск Recommendation System API...")
    print(f"🌍 Хост: {host}")
    print(f"🔗 Порт: {port}") 
    print(f"🏷️ Окружение: {environment}")
    print(f"📁 Логи: {log_dir}/api.log")
    print("📚 Документация: http://localhost:8001/docs")  # Обновлен порт в URL
    print("💾 Кеширование: In-Memory (улучшенное)")
    print("🔓 CORS: Разрешены все домены")
    print("🎯 Готов к работе!")
    
    try:
        uvicorn.run(
            app, 
            host=host, 
            port=port, 
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"❌ Ошибка запуска сервера: {e}")
        print(f"❌ Не удалось запустить сервер: {e}")
        print("💡 Попробуйте изменить порт в настройках (PORT=8002)")