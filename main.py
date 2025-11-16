"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SimpleRLDataContainer:
    """
    Класс для загрузки и предобработки данных.
    Отвечает за чтение CSV файлов, создание признаков и подготовку данных для модели.
    """
    
    def __init__(self, data_path="data"):
        # Инициализация путей к данным и структуры признаков
        self.data_path = data_path
        self.train_path = f"{data_path}/train.csv"
        self.test_path = f"{data_path}/test.csv"
        
        # Данные будут храниться здесь после загрузки
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.X_test = None
        self.actions = None
        self.rewards = None
        
        # Классификация признаков по типам
        self.numeric_features = ['recency', 'history']  # Числовые признаки
        self.binary_features = ['mens', 'womens', 'newbie']  # Бинарные признаки (0/1)
        self.categorical_features = ['zip_code', 'channel']  # Категориальные признаки
        
        # Инструменты для предобработки
        self.scaler = StandardScaler()  # Для нормализации числовых признаков
        
        # Сопоставление названий действий с числовыми кодами
        self.action_mapping = {
            'Mens E-Mail': 0,
            'Womens E-Mail': 1, 
            'No E-Mail': 2
        }
        
    def load_data(self):
        """Загрузка тренировочных и тестовых данных из CSV файлов"""
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        return self
    
    def prepare_features(self):
        """Основной метод подготовки признаков"""
        # Числовые признаки
        X_train_numeric = self.train_df[self.numeric_features].copy()
        X_test_numeric = self.test_df[self.numeric_features].copy()
        
        # Бинарные признаки
        X_train_binary = self.train_df[self.binary_features].copy()
        X_test_binary = self.test_df[self.binary_features].copy()
        
        # CTR encoding для категориальных признаков
        # Заменяем категории на среднюю вероятность визита для этой категории
        ctr_maps = {}
        for col in self.categorical_features:
            ctr_map = {}
            for value in self.train_df[col].unique():
                mask = self.train_df[col] == value
                ctr = self.train_df.loc[mask, 'visit'].mean()
                ctr_map[value] = ctr
            ctr_maps[col] = ctr_map
        
        # Применяем CTR encoding к тренировочным данным
        X_train_ctr = self.train_df[self.categorical_features].copy()
        for col in self.categorical_features:
            X_train_ctr[col] = X_train_ctr[col].map(ctr_maps[col])
        
        # Применяем CTR encoding к тестовым данным
        X_test_ctr = self.test_df[self.categorical_features].copy()
        for col in self.categorical_features:
            X_test_ctr[col] = X_test_ctr[col].map(ctr_maps[col])
        
        # Создаем дополнительные "умные" признаки
        X_train_smart = self._create_smart_features(self.train_df)
        X_test_smart = self._create_smart_features(self.test_df)
        
        # Объединяем все признаки в финальные матрицы
        self.X_train = pd.concat([X_train_numeric, X_train_binary, X_train_ctr, X_train_smart], axis=1)
        self.X_test = pd.concat([X_test_numeric, X_test_binary, X_test_ctr, X_test_smart], axis=1)
        
        # Нормализация числовых признаков
        cols_to_scale = self.X_train.columns
        self.X_train[cols_to_scale] = self.scaler.fit_transform(self.X_train[cols_to_scale])
        self.X_test[cols_to_scale] = self.scaler.transform(self.X_test[cols_to_scale])
        
        # Подготовка целевых переменных
        self.actions = self.train_df['segment'].map(self.action_mapping).values
        self.rewards = self.train_df['visit'].values
        
        return self
    
    def _create_smart_features(self, df):
        """Создание дополнительных engineered features"""
        features = pd.DataFrame(index=df.index)
        # Взаимодействие признаков
        features['recency_x_history'] = df['recency'] * df['history']
        # Логарифмирование для нормализации распределения
        features['log_history'] = np.log(df['history'] + 1)
        # Общий уровень интереса к товарам
        features['total_interest'] = df['mens'] + df['womens']
        # Доминирующий интерес
        features['dominant_mens'] = (df['mens'] > df['womens']).astype(int)
        features['dominant_womens'] = (df['womens'] > df['mens']).astype(int)
        return features
    
    def get_train_data(self):
        """Возвращает подготовленные тренировочные данные"""
        return self.X_train.values, self.actions, self.rewards
    
    def get_test_data(self):
        """Возвращает подготовленные тестовые данные"""
        return self.X_test.values
    
    def get_test_ids(self):
        """Возвращает идентификаторы тестовых клиентов"""
        return self.test_df['id'].values

class ImprovedSmartHybridEpsilonGreedy:
    """
    Гибридная модель, сочетающая Reinforcement Learning (RL) и Machine Learning (ML).
    RL обеспечивает надежность, ML - точность для конкретных случаев.
    """
    
    def __init__(self, n_arms=3, epsilon=0.05):
        # Основные параметры
        self.n_arms = n_arms  # Количество возможных действий
        self.epsilon = epsilon  # Вероятность исследования (exploration)
        
        # ML компонент: отдельная модель для каждого действия
        self.ml_models = [None, None, None]
        
        # RL компонент: статистика по эффективности действий
        self.arm_rewards = np.zeros(n_arms)  # Средняя награда для каждого действия
        self.arm_counts = np.zeros(n_arms)   # Количество использований каждого действия
        self.arm_confidence = np.zeros(n_arms)  # Уверенность в оценке каждого действия
        self.best_action = 0  # Лучшее действие по RL статистике
        
        # Для удобства интерпретации
        self.action_names = ['Mens Email', 'Womens Email', 'No Email']
        
        # Для воспроизводимости результатов
        np.random.seed(42)
    
    def fit(self, X, actions, rewards):
        """Основной метод обучения модели"""
        print("ОБУЧЕНИЕ IMPROVED SMART HYBRID MODEL...")
        print("=" * 50)
        
        # 1. Сбор RL статистики (общая эффективность действий)
        print("СОБИРАЕМ RL СТАТИСТИКУ...")
        self._collect_robust_rl_statistics(actions, rewards)
        
        # 2. Обучение ML моделей (индивидуальные предсказания)
        print("ОБУЧАЕМ ML МОДЕЛИ...")
        self._train_improved_ml_models(X, actions, rewards)
        
        print("\nОБУЧЕНИЕ ЗАВЕРШЕНО!")
        return self
    
    def _collect_robust_rl_statistics(self, actions, rewards):
        """Сбор статистики по эффективности каждого действия"""
        # Обновляем статистику для каждого наблюдения
        for action, reward in zip(actions, rewards):
            self.arm_counts[action] += 1
            # Инкрементальное обновление средней награды
            self.arm_rewards[action] += (reward - self.arm_rewards[action]) / self.arm_counts[action]
        
        # Вычисляем уверенность для каждого действия (чем больше данных, тем выше уверенность)
        for action in range(3):
            self.arm_confidence[action] = min(1.0, np.log(self.arm_counts[action] + 1) / 10)
        
        # Определяем лучшее действие
        self.best_action = np.argmax(self.arm_rewards)
        
        # Вывод статистики
        print("   Итоговая статистика:")
        for action in range(3):
            confidence = self.arm_confidence[action]
            print(f"     {self.action_names[action]}: "
                  f"награда={self.arm_rewards[action]:.3f}, "
                  f"выборов={self.arm_counts[action]:.0f}, "
                  f"уверенность={confidence:.3f}")
        
        best_reward = self.arm_rewards[self.best_action]
        print(f"   Лучшее действие: {self.action_names[self.best_action]} "
              f"(награда: {best_reward:.3f})")
        
        # Анализ преимущества лучшего действия над вторым
        rewards_sorted = sorted(self.arm_rewards, reverse=True)
        advantage = rewards_sorted[0] - rewards_sorted[1] if len(rewards_sorted) > 1 else 0
        print(f"   Преимущество над вторым: {advantage:.3f}")
    
    def _train_improved_ml_models(self, X, actions, rewards):
        """Обучение ML моделей для каждого действия"""
        for action in range(3):
            # Выбираем данные только для текущего действия
            action_mask = (actions == action)
            X_action = X[action_mask]
            y_action = rewards[action_mask]
            
            print(f"   Действие {action} ({self.action_names[action]}):")
            print(f"     Примеров: {len(X_action)}, Успехов: {np.sum(y_action)}")
            
            # Обучаем модель только если достаточно данных
            if len(X_action) > 50 and len(np.unique(y_action)) > 1:
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=25,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_action, y_action)
                self.ml_models[action] = model
                
                # Оценка качества модели
                train_score = model.score(X_action, y_action)
                print(f" Обучена! Accuracy: {train_score:.3f}")
    
    def predict_proba(self, X):
        """Основной метод предсказания вероятностей действий"""
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_arms))
        
        print(f"ПРЕДСКАЗЫВАЕМ ДЛЯ {n_samples} КЛИЕНТОВ...")
        print("   Стратегия: RL основной + ML коррекция + МИНИМАЛЬНОЕ исследование")
        
        # Для каждого клиента принимаем решение
        for i in range(n_samples):
            x = X[i]
            
            # Получаем предсказания от ML моделей
            ml_predictions = self._get_ml_predictions(x)
            
            # Начинаем с RL лучшего действия
            base_probs = np.zeros(self.n_arms)
            base_probs[self.best_action] = 1.0
            
            # Принимаем финальное решение
            final_probs = self._rl_prioritized_decision(base_probs, ml_predictions, i)
            
            probabilities[i] = final_probs
        
        return probabilities
    
    def _get_ml_predictions(self, x):
        """Получение предсказаний от всех ML моделей"""
        predictions = []
        for action in range(3):
            if self.ml_models[action] is not None:
                # Предсказываем вероятность визита для данного действия
                prob = self.ml_models[action].predict_proba([x])[0][1]
                predictions.append(prob)
            else:
                predictions.append(0.0)
        return np.array(predictions)
    
    def _rl_prioritized_decision(self, base_probs, ml_predictions, client_idx):
        """Принятие решения с приоритетом RL"""
        
        # С вероятностью epsilon идем на исследование
        if np.random.random() < self.epsilon:
            return self._conservative_exploration(ml_predictions)
        else:
            # В остальных случаях используем знания
            return self._conservative_exploitation(base_probs, ml_predictions, client_idx)
    
    def _conservative_exploration(self, ml_predictions):
        """Консервативная стратегия исследования"""
        # Следуем ML предсказаниям, но с осторожностью
        if np.sum(ml_predictions) > 0:
            ml_probs = ml_predictions / np.sum(ml_predictions)
        else:
            ml_probs = np.ones(self.n_arms) / self.n_arms
        
        # Смешиваем с равномерным распределением для безопасности
        uniform_probs = np.ones(self.n_arms) / self.n_arms
        final_probs = 0.7 * ml_probs + 0.3 * uniform_probs
        
        # Гарантируем корректность вероятностей
        final_probs = np.maximum(final_probs, 0.01)
        final_probs = final_probs / np.sum(final_probs)
        
        return final_probs
    
    def _conservative_exploitation(self, base_probs, ml_predictions, client_idx):
        """Консервативное использование знаний с ML коррекцией"""
        # Начинаем с RL лучшего действия
        final_probs = base_probs.copy()
        
        # Проверяем условия для ML коррекции
        ml_best_action = np.argmax(ml_predictions)
        
        if (ml_best_action != self.best_action and 
            self.ml_models[ml_best_action] is not None):
            
            # Вычисляем преимущество ML над RL
            ml_advantage = ml_predictions[ml_best_action] - ml_predictions[self.best_action]
            
            # Строгие условия для переключения на ML
            if ml_advantage > 0.15:
                # Случайное решение для разнообразия
                if np.random.random() < 0.2:
                    final_probs = np.zeros(self.n_arms)
                    final_probs[ml_best_action] = 1.0
                    
                    # Логируем только первые несколько решений
                    if client_idx < 2:
                        print(f"     Клиент {client_idx + 1}: "
                              f"ML переопределил RL (advantage: {ml_advantage:.3f})")
        
        return final_probs

class SNIPSEvaluator:
    """
    Класс для оценки качества политики с помощью SNIPS метрики.
    SNIPS (Self-Normalized Inverse Propensity Scoring) корректирует смещение
    в данных, вызванное тем, что исторические данные собирались другой политикой.
    """
    
    def __init__(self, logging_policy_proba=1/3):
        # Вероятность выбора действия исторической политикой
        self.logging_policy_proba = logging_policy_proba
    
    def evaluate_policy(self, actions, rewards, policy_probas):
        """Оценка политики с помощью SNIPS метрики"""
        
        weights = []
        for i, action in enumerate(actions):
            # Вероятность выбора этого действия нашей политикой
            pi_a_given_x = policy_probas[i, action]
            # Вес = наша политика / историческая политика
            weight = pi_a_given_x / self.logging_policy_proba
            weights.append(weight)
        
        weights = np.array(weights)
        
        # SNIPS оценка (взвешенное среднее)
        numerator = np.sum(weights * rewards)
        denominator = np.sum(weights)
        snips_value = numerator / denominator if denominator != 0 else 0
        
        # Лучшая статическая политика (всегда одно действие)
        static_values = []
        for action in [0, 1, 2]:
            mask = (actions == action)
            if np.sum(mask) > 0:
                static_value = np.mean(rewards[mask])
                static_values.append(static_value)
        
        best_static = max(static_values) if static_values else 0
        
        # Финальный скор: насколько наша политика лучше лучшей статической
        score = snips_value - best_static
        
        return score, snips_value, best_static


def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """

    # Создать пандас таблицу submission
    submission = predictions

    import os
    import pandas as pd
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path


def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Загрузка и подготовка данных
    print("загрузка данных...")
    data_container = SimpleRLDataContainer()
    data_container.load_data()
    data_container.prepare_features()
    
    X_train, actions, rewards = data_container.get_train_data()
    X_test = data_container.get_test_data()
    test_ids = data_container.get_test_ids()
    
    print(f"   Тренировочные данные: {X_train.shape}")
    print(f"   Тестовые данные: {X_test.shape}")
    print(f"   Всего наград (visit=1): {np.sum(rewards)}/{len(rewards)}")
    
    # Обучение модели
    print("\nОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ...")
    model = ImprovedSmartHybridEpsilonGreedy(n_arms=3, epsilon=0.05)
    model.fit(X_train, actions, rewards)
    
    # Оценка качества на части тренировочных данных
    print("\nОЦЕНКА КАЧЕСТВА...")
    evaluator = SNIPSEvaluator()
    train_probas = model.predict_proba(X_train[:1000])
    score, snips, best_static = evaluator.evaluate_policy(
        actions[:1000], rewards[:1000], train_probas
    )
    
    print(f"   SNIPS Value: {snips:.4f}")
    print(f"   Best Static Policy: {best_static:.4f}")
    print(f"   FINAL SCORE: {score:.4f}")
    
    # Предсказание на тестовых данных
    print("\nПРЕДСКАЗАНИЕ НА ТЕСТЕ...")
    probabilities = model.predict_proba(X_test)
    
    # Создание submission DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        'p_mens_email': probabilities[:, 0],
        'p_womens_email': probabilities[:, 1], 
        'p_no_email': probabilities[:, 2]
    })
    
    # Нормализация вероятностей (сумма = 1)
    sums = submission[['p_mens_email', 'p_womens_email', 'p_no_email']].sum(axis=1)
    submission[['p_mens_email', 'p_womens_email', 'p_no_email']] = (
        submission[['p_mens_email', 'p_womens_email', 'p_no_email']].div(sums, axis=0)
    )
    
    # Статистика submission
    print("\nСТАТИСТИКА SUBMISSION:")
    for col in ['p_mens_email', 'p_womens_email', 'p_no_email']:
        mean_val = submission[col].mean()
        std_val = submission[col].std()
        print(f"   {col}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Анализ распределения рекомендаций
    best_actions = np.argmax(probabilities, axis=1)
    for action in range(3):
        action_count = np.sum(best_actions == action)
        print(f"   {model.action_names[action]}: {action_count/len(best_actions):.1%}")
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)


if __name__ == "__main__":
    main()