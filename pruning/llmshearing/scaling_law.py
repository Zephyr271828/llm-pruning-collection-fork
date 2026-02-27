
from scipy.optimize import curve_fit
import numpy as np

def loss(model_size, data_size):
    """Returns the loss of a model with model_size parameters and data_size data points"""
    return E + A / (model_size ** alpha) + B / (data_size ** beta)

def get_data(l, model_size):
    """Returns the data size needed for a model of size model_size to achieve loss l"""
    return (B / (l - E - A / model_size ** alpha)) ** (1 / beta)

def inf_data_loss(model_size):
    """Returns the loss of a model with model_size parameters and infinite data points"""
    return E + A / (model_size ** alpha)

# Define the function you want to fit
def func(x, a, b, c):
    return a / np.power(x, b) + c

def get_data(y, a, b, c):
    return (a / (y - c)) ** (1 / b)

# model size
x = np.asarray([1498482688, 8030261248, 70553706496])

# losses
arxiv_loss = np.array([1.4363, 1.2102, 1.0647])
book_loss = np.array([2.5022, 2.1083, 1.6432])
c4_loss = np.array([2.6416, 2.2675, 2.0248])
cc_loss = np.array([2.5039, 2.1218, 1.7681])
github_loss = np.array([1.0065, 0.7472, 0.5551])
stackexchange_loss = np.array([2.0769, 1.7695, 1.4714])
wiki_loss = np.array([1.9663, 1.5645, 1.3009])

losses = {"cc": cc_loss, "github": github_loss, "book": book_loss, "stackexchange": stackexchange_loss, "wiki": wiki_loss, "arxiv": arxiv_loss, "c4": c4_loss}

# Fit the function to the data
target_model_size = 3606752256
for domain in ["arxiv", "book", "c4", "cc", "github", "stackexchange", "wiki"]:
    params, params_covariance = curve_fit(func, x, losses[domain], maxfev=20000, p0 = np.asarray([406.4, 0.336, 0.69]))
    a_fit, b_fit, c_fit = params
    print(f"Domain {domain}: {round(func(target_model_size, a_fit, b_fit, c_fit), 4)}")
    
# prediction for llama3.2-3b
# Domain arxiv: 1.3008
# Domain book: 2.2922
# Domain c4: 2.4177
# Domain cc: 2.29
# Domain github: 0.855
# Domain stackexchange: 1.9064
# Domain wiki: 1.7263

# actual losses:

