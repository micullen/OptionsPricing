import math
from scipy.stats import norm

class Parameters():
    def __init__(self, S0, St, T: float, t0: float, q, r, K, sigma):
        self.S0 = S0
        self.St = St
        self.T = T
        self.t0 = t0
        self.q = q
        self.r = r
        self.K = K
        self.sigma = sigma


def calc_F(params):
    F = params.St * math.exp((params.r - params.q) * (params.T - params.t0))
    return F


def calc_d1(params, F):
    d1_numer = math.log(F/params.K) + (params.T - params.t0) * 0.5 * params.sigma ** 2
    d1_denom = params.sigma * (params.T - params.t0) ** 0.5
    return d1_numer/d1_denom


def calc_d2(params, d1):
    d2 = d1 - params.sigma * (params.T - params.t0) ** 0.5
    return d2


def call(params):
    F = calc_F(params)
    d1 = calc_d1(params, F)
    d2 = calc_d2(params, d1)
    bs = math.exp(-params.r * (params.T - params.t0)) * (F * norm.cdf(d1) - params.K * norm.cdf(d2))
    return bs

def put(params):
    F = calc_F(params)
    d1 = calc_d1(params, F)
    d2 = calc_d2(params, d1)
    bs = math.exp(-params.r * (params.T - params.t0)) * (params.K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return bs

def delta_c(params):
    F = calc_F(params)
    d1 = calc_d1(params, F)
    bs = math.exp(-params.q * (params.T - params.t0)) * norm.cdf(d1)
    return bs

def delta_p(params):
    F = calc_F(params)
    d1 = calc_d1(params, F)
    bs = - math.exp(- params.q * (params.T - params.t0)) * norm.cdf(-d1)
    return bs

param_set = Parameters(S0=100, St=110, T=1, t0=0, q=0, r=0.2, K=120, sigma=0.3)


# Put call parity
call_val = call(param_set)
put_val = put(param_set)

# call - put = S(t) - K exp(-r (T - t))

lhs = call_val - put_val
rhs = param_set.St - param_set.K * math.exp(-param_set.r * (param_set.T - param_set.t0))

print(lhs)
print(rhs)

x = call(param_set)
y = put(param_set)

print(x)
print(y)







