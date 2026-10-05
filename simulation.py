import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

rng = np.random.default_rng(50)

# регрессионая функция
def f_theta(t, theta):
    return 0.5 * (1.0 / (1.0 + (t - theta) ** 2) +
                  1.0 / (1.0 + (t - (1.0 - theta)) ** 2))


def sigma_theta(t, theta):
    return 0.05 * (1.0 + 2.0 * np.abs(t - 0.5)) * (1.0 + theta)

# численная оценка производной
def f_theta_prime(t, theta, eps=1e-6):
    return (f_theta(t, theta + eps) - f_theta(t, theta - eps)) / (2 * eps)

# исходные данные модели
THETA0 = 0.3
N = 300
T1 = 0.5

# генерация векторов регерссеров и откликов
def simulate_sample(n=N, theta0=THETA0, rng=rng):
    z = np.arange(1, n + 1) / n
    xi = rng.standard_normal(n)
    X = f_theta(z, theta0) + sigma_theta(z, theta0) * xi
    return z, X

# ядро
def epanechnikov(u):
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    mask = np.abs(u) <= 1.0
    out[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return out

# ядерная локально-постоянная оценка
def kernel_estimator(t, z, X, h):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(z)
    lam = 1.0 / n
    diff = (t[:, None] - z[None, :]) / h
    K = epanechnikov(diff) / h
    num = (K * X[None, :] * lam).sum(axis=1)
    den = (K * lam).sum(axis=1)
    out = np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)
    return out if out.size > 1 else out[0]

# ур-ие квазиподобия 
def Sn(theta, z, X):
    s2 = sigma_theta(z, theta) ** 2
    fp = f_theta_prime(z, theta)
    return np.sum(fp / s2 * (X - f_theta(z, theta)))

# вычисление универсальной оценки параметра
def explicit_estimator(z, X, h, t1=T1, theta_init=0.3):
    f_star_t1 = kernel_estimator(t1, z, X, h)

    def diff(theta):
        return f_theta(t1, theta) - f_star_t1

    def diff_prime(theta):
        return f_theta_prime(t1, theta)

    try:
        theta_star = newton(diff, theta_init, fprime=diff_prime,
                             tol=1e-8, maxiter=100)
    except RuntimeError:
        theta_star = np.nan
    return theta_star, f_star_t1


z, X = simulate_sample()
theta_grid = np.linspace(0.0, 0.6, 400)
Sn_vals = -np.array([Sn(th, z, X) for th in theta_grid])
# вывод графика уравнения квазиподобия
plt.figure(figsize=(7, 5))
plt.axhline(0, color="red", linestyle=":", linewidth=0.8)
plt.plot(theta_grid, Sn_vals, color="blue")
plt.ylim(-200, 200)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

h = N ** (-1.0 / 3.0)
# вывод графика регрессионной функции в сравнении с ядерной оценкой и отклики
t_grid = np.linspace(0, 1, 400)
f_true = f_theta(t_grid, THETA0)
f_hat = kernel_estimator(t_grid, z, X, h)

plt.figure(figsize=(7, 5))
plt.scatter(z, X, s=8, alpha=0.4, color="tab:blue")
plt.plot(t_grid, f_true, color="green", linewidth=2)
plt.plot(t_grid, f_hat, color="red", linestyle="--", linewidth=2)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# вывод графика уравнения для поиска универсальной оценки параметра
theta_star, f_star_t1 = explicit_estimator(z, X, h, t1=T1)
diff_grid = f_theta(T1, theta_grid) - f_star_t1

plt.figure(figsize=(7, 5))
plt.axhline(0, color="red", linestyle=":", linewidth=0.8)
plt.plot(theta_grid, diff_grid, color="blue")
if not np.isnan(theta_star):
    plt.scatter([theta_star], [0], facecolors="none", edgecolors="black",
                s=60, zorder=5)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# нахождение среднего смещения оценки на нескольких симуляциях
N_SIM = 100
theta_estimates = []

for _ in range(N_SIM):
    z_i, X_i = simulate_sample(n=N, theta0=THETA0)
    th_hat, _ = explicit_estimator(z_i, X_i, h, t1=T1)
    if not np.isnan(th_hat):
        theta_estimates.append(th_hat)

theta_estimates = np.array(theta_estimates)
mean_estimate = theta_estimates.mean()
bias = mean_estimate - THETA0

print('Number of simulations', len(theta_estimates))
print('Sample size n', N)
print('True parameter theta0', THETA0)
print('Mean of estimates', round(mean_estimate,4))
print('Bias', round(bias,4))

