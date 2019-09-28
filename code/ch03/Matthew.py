#Chapter 3
#%%
import sympy

sympy.init_printing()

from sympy import I, pi, oo

#%% md
## Symbol ##
#%%
x = sympy.Symbol("x")
y = sympy.Symbol("y", real = True) #y is a real number
#%%
y.is_real

#%%
x.is_real is None #x is not known whether real number. 
#is_real return either True, False or None

#%%
sympy.Symbol("z", imaginary = True).is_real #Imaginary Number (허수)

#Symbol Objects
#real / imaginary
#positive / negative
#integer
#odd / even
#prime
#finite / infinite

#%%
x = sympy.Symbol("x")
y = sympy.Symbol("y", positive = True)
sympy.sqrt(x ** 2)
sympy.sqrt(y ** 2)

#%%
n1 = sympy.Symbol("n")
n2 = sympy.Symbol("n", integer = True)
n3 = sympy.Symbol("n", odd = True)
sympy.cos(n1 * pi)
#%%
sympy.cos(n2 * pi)
#%%
sympy.cos(n3 * pi)

#%%
a, b, c = sympy.symbols("a, b, c", negative = True)
d, e, f = sympy.symbols("d, e, f", positive = True)

#%%
i = sympy.Integer(19)
type(i)
#%%
i.is_Integer, i.is_real, i.is_odd
f = sympy.Float(2.3)
type(f)
#%%
f.is_Integer, f.is_real, f.is_odd
#%%
i, f = sympy.sympify(19), sympy.sympify(2.3)
type(i), type(f)
#%% md
## Integer ##
#%%
n = sympy.Symbol("n", integer = True)
n.is_integer, n.is_Integer, n.is_positive, n.is_Symbol #integer = integer, Integer= specific integer

#%%
i = sympy.Integer(19)
n.is_integer, n.is_Integer, n.is_positive, n.is_Symbol

#%%
i ** 50
#%%
sympy.factorial(100)
#Integer in SymPy are arbitary precision

#%% md
## Float ##

#%%
"%.25f" % 0.3
#%%
sympy.Float(0.3, 25)
#%%
sympy.Float('0.3', 25)

#%% md
## Rational ##

#%%
sympy.Rational(11, 13)

#%%
r1 = sympy.Rational(2,3)
r2 = sympy.Rational(4,5)
r1 * r2

#%%
r1 / r2

#%%
sympy.pi

#%%
sympy.E #natural logarithm

#%%
sympy.EulerGamma # Euler's Constant

#%%
sympy.I # Imaginary Unit

#%%
sympy.oo # Infinity

#%% md
## Functions ##

#%%
x, y, z = sympy.symbols("x, y, z")
f = sympy.Function("f")
type(f)

#%%
f(x)

#%%
g = sympy.Function("g")(x, y, z)
g

#%%
g.free_symbols #returns a set of unique symbols contained in a given expression. Important for derivatives of abstract func or specifying diffential equations

#%%
sympy.sin

#%%
sympy.sin(x) #abstract symbol, unevaluated

#%%
sympy.sin(pi * 1.5)

#%%
n = sympy.Symbol("n", integer = True)
sympy.sin(pi * n)

#%%
h = sympy.Lambda(x, x**2)
h

#%%
h(5)

#%%
h(1 + x)

#%% md
## Expressions ##

#%%
x = sympy.Symbol("x")
expr = 1 + 2 * x**2 + 3 * x**3
expr

#%%
from sympy.printing.dot import dotprint
from graphviz import Digraph, Source
dp = dotprint(expr)
dot = Source(dp)
dot.render('test-output/round-table.gv', view=True)  

#%%
expr.args

#%%
expr.args[1]

#%%
expr.args[1].args[1]

#%% md
## Manipulating Expressions ##

#%% md
### Simplification ###

#%%
expr = 2 * (x**2 - x) - x * (x + 1)
expr

#%%
sympy.simplify(expr)

#%%
expr.simplify()

#%%
expr

#%%
expr = 2 * sympy.cos(x) * sympy.sin(x)
expr

#%%
sympy.simplify(expr)

#%%
expr = sympy.exp(x) * sympy.exp(y)
expr

#%%
sympy.simplify(expr)

# Alternative Simplyfing Expressions
# trigsimp: simplify using trigonometric identities
# powsimp: simplify using laws of powers
# compsimp: simplify combinatorial expr
# ratsimp: simplify by writing on a common denominator

#%% md
### Expand ###

#%%
expr = (x + 1) * (x + 2)
sympy.expand(expr)

#%%
sympy.sin(x + y).expand(trig = True) #trigonometric expansion, mul=True for expanding product like above

#%%
a,b = sympy.symbols("a, b", positive=True)
sympy.log(a*b).expand(log=True) #expanding logarithms

#%%
sympy.exp(I*a + b).expand(complex=True) #real / imaginary parts of an expression

#%%
sympy.expand((a * b)**x, power_base=True) #expanding the base

#%%
sympy.exp((a-b)*x).expand(power_exp=True) #expending the exponent of a power expression

#%% md
### Factor, collect, and combine ###

#%%
sympy.factor(x**2 - 1)

#%%
sympy.factor(x * sympy.cos(y) + sympy.sin(z) * x)

#%%
sympy.logcombine(sympy.log(a) - sympy.log(b)) #combine

#%%
expr = x + y + x * y * z
expr.collect(x)

#%%
expr.collect(y)

#%%
expr = sympy.cos(x+y) + sympy.sin(x-y)
expr.expand(trig=True).collect([sympy.cos(x), sympy.sin(x)]).collect(sympy.cos(y) - sympy.sin(y))

#%% md
### Apart, together, and cancel ###

#%%
sympy.apart(1/(x**2 + 3*x + 2), x) # rewrite expr as the partial fraction

#%%
sympy.together(1 / (y * x + 1) +1 / (1+x)) # combine the sum of fractions

#%%
sympy.cancel(y / (y * x + y)) # to cancel shared factor btw 분모,분자

#%% md
### Substitutions ###

#%%
(x + y).subs(x, y)

#%%
sympy.sin(x * sympy.exp(x)).subs(x,y)

#%% md
### Numerical Evaluation ###

#%%
sympy.N(1 + pi)

#%%
sympy.N(pi, 50)

#%%
(x + 1/pi).evalf(10)

#%%
expr = sympy.sin(pi * x * sympy.exp(x))
[expr.subs(x, xx).evalf(3) for xx in range(0, 10)]

#%% md
## Calculus ##

### Derivatives ###

#%%
f = sympy.Function('f')(x)
sympy.diff(f, x)

#%%
sympy.diff(f, x, x)

#%%
sympy.diff(f, x, x, x)

#%%
g = sympy.Function('g')(x,y)
g.diff(x,y)

#%%
g.diff(x, 3, y, 2)

#%% md
### Integrals ###

#%%
a,b,x,y = sympy.symbols("a,b,x,y")
sympy.integrate(f)

#%%
sympy.integrate(f, (x, a, b))

#%% md
### Series ###

#%%
x, y = sympy.symbols("x, y")
f = sympy.Function("f")(x)
sympy.series(f, x)

#%% md
### Limits ###

#%%
sympy.limit(sympy.sin(x) / x, x, 0)