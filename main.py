import numpy as np
import matplotlib.pyplot as plt


# Определяем гауссово ядро:
def gaussian_kernel(x, y, sigma=1.0):
    return np.exp(- (x - y) ** 2 / (2 * sigma ** 2))


# Для примера можно определить и sinc (если необходимо):
def sinc(x):
    return np.sinc(x / np.pi)  # np.sinc реализует sin(pi*x)/(pi*x)


# Реализация обновления KRLS для одного шага.
# Аргументы:
#   x_t, y_t: текущий вход и значение функции
#   D: текущий массив (numpy.ndarray) опорных точек (1D)
#   K_inv: текущая инвертированная матрица по D (размера m x m)
#   alpha: текущие коэффициенты (вектор длины m)
#   P: матрица P из алгоритма (размера m x m)
#   sigma: параметр ядра
#   nu: порог точности аппроксимации (ALD)
#
# Функция возвращает обновлённые D, K_inv, alpha, P, а также величину delta и булев флаг,
# добавилась ли новая точка (True = новый элемент добавлен).
def krls_update(x_t, y_t, D, K_inv, alpha, P, sigma, nu):
    m = len(D)
    # Вычисляем вектор ядерных значений между x_t и всеми опорными точками
    k_tilde = np.array([gaussian_kernel(x_d, x_t, sigma) for x_d in D])
    k_tt = gaussian_kernel(x_t, x_t, sigma)
    # Вычисляем a_t = K_inv @ k_tilde
    a_t = K_inv.dot(k_tilde)
    # delta = k(x_t, x_t) - k_tilde^T * a_t
    delta = k_tt - k_tilde.dot(a_t)

    # Если delta > nu, новая точка считается линейно независимой в аппроксимированном пространстве -> добавляем
    if delta > nu:
        # Вычисляем ошибку предсказания для текущей точки по старой модели
        e_t = y_t - k_tilde.dot(alpha)
        # Обновление инвертированной матрицы: формирование нового блочного обратного:
        new_m = m + 1
        new_K_inv = np.zeros((new_m, new_m))
        # Верхний левый блок:
        new_K_inv[:m, :m] = K_inv + np.outer(a_t, a_t) / delta
        # Правый столбец:
        new_K_inv[:m, m] = -a_t / delta
        # Нижняя строка:
        new_K_inv[m, :m] = -a_t / delta
        new_K_inv[m, m] = 1.0 / delta

        # Обновляем коэффициенты: расширяем вектор alpha
        # Согласно (3.16):
        new_alpha = np.concatenate([alpha - a_t * (e_t / delta), [e_t / delta]])
        # Обновление матрицы P: согласно (3.15) просто расширяем до блочной диагонали
        new_P = np.zeros((new_m, new_m))
        new_P[:m, :m] = P
        new_P[m, m] = 1.0

        # Обновляем словарь: добавляем новую точку
        D_new = np.append(D, x_t)
        added = True
        return D_new, new_K_inv, new_alpha, new_P, delta, added
    else:
        # Если точка аппроксимируется существующим словарём (delta <= nu)
        e_t = y_t - k_tilde.dot(alpha)
        # Согласно (3.12): обновление матрицы P
        tmp = P.dot(a_t)
        denominator = 1 + a_t.dot(tmp)
        P_new = P - np.outer(tmp, tmp) / denominator
        # Согласно (3.13): обновление alpha
        # Здесь используется текущая K_inv (не изменяется, т.к. словарь не расширяется)
        update_term = K_inv.dot(tmp) * e_t
        alpha_new = alpha + update_term
        added = False
        return D, K_inv, alpha_new, P_new, delta, added


# Функция предсказания: f(x) = sum_j alpha_j * k(x_j, x)
def predict(x, D, alpha, sigma):
    # Гарантируем, что x – массив хотя бы 1D
    x = np.atleast_1d(x)
    pred = np.array([np.sum(alpha * np.array([gaussian_kernel(x_d, xi, sigma) for x_d in D]))
                     for xi in x])
    return pred



# Основная процедура: обучение KRLS на l точках
def train_krls(x_train, y_train, sigma=1.0, nu=1e-3):
    # Инициализация: первая точка формирует словарь
    D = np.array([x_train[0]])
    # Для первой точки k(x1,x1)=1 (при нормировке ядра) или exp(0)=1
    K_inv = np.array([[1.0]])
    # Начальные коэффициенты: alpha = [y1] (так как f(x1)=y1)
    alpha = np.array([y_train[0]])
    # Инициализация матрицы P
    P = np.array([[1.0]])

    # Для хранения истории: будем сохранять ошибку, число опорных векторов и, возможно, величину delta
    n_points = len(x_train)
    support_vec_count = [len(D)]
    mse_history = []  # можно сохранить ошибки предсказания на обучающей выборке по мере обучения

    # Для каждого нового сэмпла начинаем обновление
    for t in range(1, n_points):
        x_t = x_train[t]
        y_t = y_train[t]
        D, K_inv, alpha, P, delta, added = krls_update(x_t, y_t, D, K_inv, alpha, P, sigma, nu)
        support_vec_count.append(len(D))
        # Можно вычислить ошибку на обучающем сэмпле (например, квадрат ошибки текущей точки)
        mse_history.append((y_t - predict(x_t, D, alpha, sigma)) ** 2)
    return D, alpha, K_inv, P, support_vec_count, mse_history


# Генерация данных и эксперимент
def run_experiment(func, noise_std=0.0, l=200, sigma=0.5, nu=1e-3):
    # Задаем диапазон для x и генерируем обучающие данные
    x_train = np.linspace(-5, 5, l)
    y_true = func(x_train)
    if noise_std > 0:
        y_train = y_true + np.random.normal(0, noise_std, size=x_train.shape)
    else:
        y_train = y_true.copy()

    # Обучаем алгоритм KRLS
    D, alpha, K_inv, P, support_vec_count, mse_history = train_krls(x_train, y_train, sigma=sigma, nu=nu)

    # Тестовые данные
    x_test = np.linspace(-5, 5, 400)
    y_pred = predict(x_test, D, alpha, sigma)
    y_test_true = func(x_test)

    # Среднеквадратичная ошибка на тестовой выборке
    mse = np.mean((y_test_true - y_pred) ** 2)

    # Возвращаем все значения, включая mse_history
    return x_train, y_train, x_test, y_test_true, y_pred, len(D), mse, support_vec_count, mse_history


# Запускаем эксперимент для точных данных и зашумлённых
np.random.seed(42)


# Можно выбрать функцию: здесь sin(x)
def func(x):
    return np.sin(x)


# Эксперимент 1: точные данные
x_train, y_train, x_test, y_test_true, y_pred, n_support, mse, support_history, mse_history = run_experiment(func, noise_std=0.0, l=200, sigma=0.8, nu=1e-3)

print("Точные данные:")
print("Кол-во опорных векторов =", n_support)
print("MSE на тестовой выборке =", mse)

# Эксперимент 2: зашумленные данные
x_train_n, y_train_n, x_test_n, y_test_true_n, y_pred_n, n_support_n, mse_n, support_history_n, mse_history_n = run_experiment(func,
                                                                                                                noise_std=0.2,
                                                                                                                l=200,
                                                                                                                sigma=0.8,
                                                                                                                nu=1e-3)
print("\nЗашумленные данные (std=0.2):")
print("Кол-во опорных векторов =", n_support_n)
print("MSE на тестовой выборке =", mse_n)

# Визуализация результатов:
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. График функции для точных данных
axs[0, 0].plot(x_test, y_test_true, label="Истинная функция", color="green", linewidth=2)
axs[0, 0].scatter(x_train, y_train, label="Обучающие точки", color="blue", s=5)
axs[0, 0].plot(x_test, y_pred, label="KRLS предсказание", color="red", linestyle="--")
axs[0, 0].set_title("Точные данные")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. График функции для зашумлённых данных
axs[0, 1].plot(x_test_n, y_test_true_n, label="Истинная функция", color="green", linewidth=2)
axs[0, 1].scatter(x_train_n, y_train_n, label="Обучающие точки", color="blue", s=5)
axs[0, 1].plot(x_test_n, y_pred_n, label="KRLS предсказание", color="red", linestyle="--")
axs[0, 1].set_title("Зашумленные данные (std=0.2)")
axs[0, 1].legend()
axs[0, 1].grid(True)


# 3. График зависимости числа опорных векторов от номера обучающей выборки (на примере точных данных)
axs[1, 0].plot(np.arange(1, len(support_history) + 1), support_history, marker='o')
axs[1, 0].set_title("Эволюция числа опорных векторов (точные данные)")
axs[1, 0].set_xlabel("Номер обучающей точки")
axs[1, 0].set_ylabel("Число опорных векторов")
axs[1, 0].grid(True)

# 4. График накопленной MSE по обучающей выборке (можно усреднить по окну)
if len(mse_history) > 0:
    cum_mse = np.cumsum(mse_history) / np.arange(1, len(mse_history) + 1)
    axs[1, 1].plot(np.arange(2, len(cum_mse) + 2), cum_mse, marker='o')
    axs[1, 1].set_title("Накопленная MSE (точные данные)")
    axs[1, 1].set_xlabel("Номер обучающей точки")
    axs[1, 1].set_ylabel("Средний квадрат ошибки")
    axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

