import sympy
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg as la
from scipy import optimize

sympy.init_printing()

def gen_equation_mat(gen):
    A = gen([[2,3], [5,4]])
    b = gen([4,3])

    return A, b

def eg1_sympy():
    A, b = gen_equation_mat(sympy.Matrix)
    cond = A.condition_number()
    n = sympy.N(cond)

    return A.rank(), cond, n, A.norm()

def eg1_numpy():
    A, b = gen_equation_mat(np.array)

    return np.linalg.matrix_rank(A), np.linalg.cond(A), np.linalg.norm(A)


# LU factorization
def LUdecomp_solution():
    A, b = gen_equation_mat(sympy.Matrix)

    L, U, _ = A.LUdecomposition()
    x = A.LUsolve(b)
    return L, U, L*U, x

def lu_solution():
    A, b = gen_equation_mat(np.array)
    P, L, U = la.lu(A)
    comp = P.dot(L.dot(U)) #A = PLU
    x = la.solve(A, b)

    return L, U, comp, x

# Large condition number
def symbolic_approach():
    p = sympy.symbols("p", positive = True)
    A = sympy.Matrix([[1, sympy.sqrt(p)], [1, 1/sympy.sqrt(p)]])
    b = sympy.Matrix([1,2])

    x = A.solve(b)
    Acond = A.condition_number().simplify()

    return p, x, Acond

def numpy_approach():
    b = np.array([1,2])
    def f(p):
        A = np.array([[1, np.sqrt(p)], [1, 1/np.sqrt(p)]])
        return np.linalg.solve(A, b)
    return f

def graph_diff():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    p_vec = np.linspace(0.9, 1.1, 200)
    
    p, x_sym_sol, Acond = symbolic_approach()
    x_num_sol = numpy_approach()
    for n in range(2):
        x_sym = np.array([x_sym_sol[n].subs(p, pp).evalf() for pp in p_vec])
        x_num = np.array([x_num_sol(pp)[n] for pp in p_vec])
        axes[0].plot(p_vec, (x_num - x_sym)/x_sym, 'k')
    axes[0].set_title("Error in solution\n(numerical - symbolic)/symbolic")
    axes[0].set_xlabel(r'$p$', fontsize=18)

    axes[1].plot(p_vec, [Acond.subs(p, pp).evalf() for pp in p_vec])
    axes[1].set_title("Condition Number")
    axes[1].set_xlabel(r'$p$', fontsize=18)
    plt.show()

def linear_least_square(repeat=2):
    x = np.linspace(-1, 1, 100)
    a, b, c = 1, 2, 3
    y_exact = a + b * x + c * x**2

    m = 100
    X = 1 - 2 * np.random.rand(m)
    Y = a + b * X + c * X**2 + np.random.randn(m)

    A = np.vstack([X**n for n in range(repeat)])
    sol, r, rank, sv = la.lstsq(A.T, Y)
    
    #y_fit = sol[0] + sol[1] * x + sol[2] * x**2
    y_fit = sum([s * x**n for n, s in enumerate(sol)])

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(X, Y, 'go', alpha=0.5, label='Simulated data')
    ax.plot(x, y_exact, 'k', lw=2, label='True value $y = 1 + 2x + 3x^2$')
    ax.plot(x, y_fit, 'b', lw=2, label='Least square fit')
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18)
    ax.legend(loc=2)
    plt.show()

def sympy_eigen():
    eps, delta = sympy.symbols("epsilon, Delta")
    H = sympy.Matrix([[eps, delta], [delta, -eps]])
    
    return H.eigenvals(), H.eigenvects()

def numpy_eigen():
    A = np.array([[1,3,5], [3,5,3], [5,3,9]])
    evals, evecs = la.eig(A)
    return la.eigvalsh(A)

def nonlinear_soln():
    x, a, b, c = sympy.symbols("x, a, b, c")
    #sympy.solve(a * sympy.con(x) - b * sympy.sin(x), x)
    return sympy.solve(a + b*x + c*x**2, x)

def nonlinear_soln2():
    x = np.linspace(-2, 2, 1000)
    f1 = x**2 - x - 1
    f2 = x**3 - 3 * np.sin(x)
    f3 = np.exp(x) - 2
    f4 = 1 - x**2 + np.sin(50 / (1 + x**2))

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for n, f in enumerate([f1, f2, f3, f4]):
        axes[n].plot(x, f, lw=1.5)
        axes[n].axhline(0, ls=':', color='k')
        axes[n].set_ylim(-5, 5)
        axes[n].set_xticks([-2, -1, 0, 1, 2])
        axes[n].set_xlabel(r'$x$', fontsize=18)
    
    axes[0].set_ylabel(r'$f(x)$', fontsize=18)

    titles = [r'$f(x)=x^2-x-1$', r'$f(x)=x^3-3\sin(x)$',
             r'$f(x)=\exp(x)-2$', r'$f(x)=\sin\left(50/(1+x^2)\right)+1-x^2$']

    for n, title in enumerate(titles):
        axes[n].set_title(title)
    plt.show()

def root_bisection():
    def f(x):
        return np.exp(x) - 2

    tol = 0.1
    a, b = -2, 2
    x = np.linspace(-2.1, 2.1, 1000)

    fig, ax = plt.subplots(1, 1, figsize=(12,4))

    ax.plot(x, f(x), lw=1.5)
    ax.axhline(0, ls=':', color='k')
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.set_ylabel(r'$f(x)$', fontsize=18)

    fa, fb = f(a), f(b)

    ax.plot(a, fa, 'ko')
    ax.plot(b, fb, 'ko')
    ax.text(a, fa + 0.5, r"$a$", ha='center', fontsize=18)
    ax.text(b, fb + 0.5, r"$b", ha='center', fontsize=18)

    n = 1
    while b - a > tol:
        m = a + (b - a) / 2
        fm = f(m)

        ax.plot(m, fm, 'ko')
        ax.text(m, fm - 0.5, r"$m_%d" % n, ha='center')
        n += 1

        if np.sign(fa) == np.sign(fm):
            a, fa = m, fm
        else:
            b, fb = m, fm

    ax.plot(m, fm, 'r*', markersize=10)
    ax.annotate("Root approximately at %.3f" % m,
                fontsize=14, family='serif',
                xy=(a, fm), xycoords='data',
                xytext=(-150, +50), textcoords='offset points',
                arrowprops=dict(arrowstyle='->',
                connectionstyle="arc3, rad=-.5"))
    ax.set_title("Bisection method")
    plt.show()

def root_newton():
    tol = 0.01
    xk = 2

    s_x = sympy.symbols("x")
    s_f = sympy.exp(s_x) - 2

    f = lambda x: sympy.lambdify(s_x, s_f, 'numpy')(x)
    fp = lambda x: sympy.lambdify(s_x, sympy.diff(s_f, s_x), 'numpy')(x)

    x = np.linspace(-1, 3.1, 1000)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(x, f(x))
    ax.axhline(0, ls=':', color='k')

    n = 0
    while f(xk) > tol:
        xk_new = xk - f(xk) / fp(xk)

        ax.plot([xk, xk], [0, f(xk)], color='k', ls=':')
        ax.plot(xk, f(xk), 'ko')
        ax.text(xk, -.5, r'$x_%d$' % n, ha='center')
        ax.plot([xk, xk_new], [f(xk), 0], 'k-')

        xk = xk_new
        n += 1

    ax.plot(xk, f(xk), 'r*', markersize=15)
    ax.annotate("Root approximately at %.3f" % xk,
             fontsize=14, family="serif",
             xy=(xk, f(xk)), xycoords='data',
             xytext=(-150, +50), textcoords='offset points',
             arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3, rad=-.5"))

    ax.set_title("Newton's method")
    ax.set_xticks([-1, 0, 1, 2])
    plt.show()

def f(x):
    return [x[1] - x[0]**3 - 2*x[0]**2 + 1, x[1] + x[0]**2 -1]

def fsolv_eg():
    def f_jacobian(x):
        return [[-4*x[0]**2-4*x[0], 1], [2*x[0], 1]]
    x, y = sympy.symbols("x, y")
    f_mat = sympy.Matrix([y - 2*x**2 + 1, y + x**2 - 1])
    return optimize.fsolve(f, [1, 1], fprime=f_jacobian)

def fsolve_eg2():
    x = np.linspace(-3, 2, 5000)
    y1 = x**3 + 2 * x**2 - 1
    y2 = -x**2 + 1

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, y1, 'b', lw=1.5, label=r'$y = x^3 + 2x^2 - 1$')
    ax.plot(x, y2, 'g', lw=1.5, label=r'$y = -x^2 + 1$')

    x_guesses = [[-2, 2], [1, -1], [-2, -5]]
    for x_guess in x_guesses:
        sol = optimize.fsolve(f, x_guess)
        ax.plot(sol[0], sol[1], 'r*', markersize=15)
        ax.plot(x_guess[0], x_guess[1], 'ko')
        ax.annotate("", xy=(sol[0], sol[1]), xytext=(x_guess[0], x_guess[1]),
                    arrowprops=dict(arrowstyle="->", linewidth=2.5))
    ax.legend(loc=0)
    ax.set_xlabel(r'$x$', fontsize=18)
    plt.show()
