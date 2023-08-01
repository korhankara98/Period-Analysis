import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import scipy as sc
import pandas as pd
import astropy as ast
import lmfit.models as lm


#1#################################################################################################################################


data = #csv

Epoch = data['Epoch']
OC = data['OC']
OC_err = data['OC_err']


#2#################################################################################################################################


plt.figure(figsize=(10,6))
plt.errorbar(Epoch, OC, yerr=OC_err, fmt='o', label='Data')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C')
plt.title("O-C with Error Bars")
plt.show()


#3#################################################################################################################################

median=np.median(OC)                                   # Ortanca Değer
Q1 = np.percentile(OC, 25)
Q3 = np.percentile(OC, 75)
IQR = Q3 - Q1

upper_bound = OC[OC <= Q3 + 1.5*IQR].max()             # IQR çarpanını 1.5
lower_bound = OC[OC >= Q1 - 1.5*IQR].min()
outliers = (OC < lower_bound) | (OC > upper_bound)     # Aykırı noktalar

print("Median: {:.2f}, Q1: {:.2f}, Q3: {:.2f}".format(median,Q1,Q3))
print("IQR: {:.2f}, Upper Bound: {:.2f}, Lower Bound: {:.2f}".\
      format(IQR,upper_bound,lower_bound))

filtered_data = data[~outliers]

T2 = filtered_data['Epoch']
OC2 = filtered_data['OC']
OC_err2 = filtered_data['OC_err']

plt.figure(figsize=(10,6))
plt.errorbar(T2, OC2, yerr=OC_err2, fmt='o', label='Data')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C')
plt.title("O-C with Error Bars (Outliers Removed)")
plt.show()


#4#################################################################################################################################


#Lineer Model

model = lm.LinearModel()
params_linear = model.guess(OC2, x=T2)
result_linear = model.fit(OC, params_linear, x=T2)
print(result_linear.fit_report())                                             #6.1


plt.figure(figsize=(10,6))
plt.errorbar(T2, OC2, yerr=OC_err2, fmt='o', label='Data')
plt.plot(T2, result_linear.best_fit, 'r-', label='Fitted Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C')
plt.title("O-C with Linear Fit")
plt.show()

# 5.1
residuals_linear = OC2 - result_linear.best_fit
errors = OC_err2

plt.figure(figsize=(10,6))
plt.errorbar(T2, residuals_linear, yerr=errors, fmt='o', label='Residuals')
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C$_{Residuals}$')
plt.title("Linear Residuals")
plt.show()



#Parabolik Model

model = lm.QuadraticModel()
params_parabolic = model.guess(OC2, x=T2)
result_parabolic = model.fit(OC2, params_parabolic, x=T2)
print(result_parabolic.fit_report())                                          #6.2


plt.figure(figsize=(10,6))
plt.errorbar(T2, OC2, yerr=OC_err2, fmt='o', label='Data')
plt.plot(T2, result_parabolic.best_fit, 'r-', label='Fitted Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C')
plt.title("O-C with Parabolic Fit")
plt.show()

# 5.2
residuals_parabolic = OC2 - result_parabolic.best_fit
errors = OC_err2

plt.figure(figsize=(10,6))
plt.errorbar(T2, residuals_parabolic, yerr=errors, fmt='o', label='Residuals')
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C$_{Residuals}$')
plt.title("Parabolic Residuals")
plt.show()



#Çevrimsel Model

def cevrimsel_model(x, A, P):
    return A * np.sin((np.radians(x)/P)*360)

model = lm.Model(cevrimsel_model)
params_cevrim = model.make_params(A=140, P=1200) 
result_cevrimsel = model.fit(OC2, params_cevrim, x=T2)
print(result_cevrimsel.fit_report())                                          #6.3


plt.figure(figsize=(10,6))
plt.errorbar(T2, OC2, yerr=OC_err2, fmt='o', label='Data')
plt.plot(T2, result_cevrimsel.best_fit, 'r-', label='Fitted Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C')
plt.title("O-C with Cyclic Fit")
plt.show()

# 5.3
residuals_cevrimsel = OC2 - result_cevrimsel.best_fit
errors = OC_err2

plt.figure(figsize=(10,6))
plt.errorbar(T2, residuals_cevrimsel, yerr=errors, fmt='o', label='Residuals')
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C$_{Residuals}$')
plt.title("Cyclic Residuals")
plt.show()



#Lineer + Çevrimsel

def linear_cevrimsel_model(x, a, b, A, P):
    return a*x + b + A * np.sin((np.radians(x)/P)*360)


model = lm.Model(linear_cevrimsel_model)
params_lc = model.make_params(a=-0.01320129, b=-53.8101412, A=23, P=980)  # a ve b değerleri lineer fonksiyon çıktısından alınmıştır.
result_lc = model.fit(OC2, params_lc, x=T2)
print(result_lc.fit_report())                                                 #6.4


plt.figure(figsize=(10,6))
plt.errorbar(T2, OC2, yerr=OC_err2, fmt='o', label='Data')
plt.plot(T2, result_lc.best_fit, 'r-', label='Fitted Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C')
plt.title("O-C with Linear+Cyclic Fit")
plt.show()

# 5.4
residuals_lc = OC2 - result_lc.best_fit
errors = OC_err2

plt.figure(figsize=(10,6))
plt.errorbar(T2, residuals_lc, yerr=errors, fmt='o', label='Residuals')
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C$_{Residuals}$')
plt.title("Linear+Cyclic Residuals")
plt.show()



#Parabolik + Çevrimsel

def parabolic_cyclic_model(x, a, b, c, A, P):
    return a*x**2 + b*x + c + A * np.sin((np.radians(x)/P)*360)


model = lm.Model(parabolic_cyclic_model)
params_pc = model.make_params(a=7.2777e-05, b=-0.01702256, c=-77.1568617, A=23, P=980) # Değerler diğer fonksiyonların çıktılarından alınmıştır.
result_pc = model.fit(OC2, params_pc, x=T2)
print(result_pc.fit_report())                                                 #6.5


plt.figure(figsize=(10,6))
plt.errorbar(T2, OC2, yerr=OC_err2, fmt='o', label='Data')
plt.plot(T2, result_pc.best_fit, 'r-', label='Fitted Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C')
plt.title("O-C with Quadratic+Cyclic Fit")
plt.show()

# 5.5
residuals_pc = OC2 - result_pc.best_fit
errors = OC_err2

plt.figure(figsize=(10,6))
plt.errorbar(T2, residuals_pc, yerr=errors, fmt='o', label='Residuals')
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C$_{Residuals}$')
plt.title("Parabolic+Cyclic Residuals")
plt.show()


#7#################################################################################################################################


# Lineer Model
RSS_linear = np.sum(residuals_linear**2)
RSS_linear_full = np.sum((OC2 - np.mean(OC2))**2)
n_linear = len(T2)
k_linear = 2  # Lineer modeldeki serbestlik derecesi

F_linear = ((RSS_linear_full - RSS_linear) / (k_linear - 0)) / (RSS_linear / (n_linear - k_linear))
AIC_linear = n_linear * np.log(RSS_linear / n_linear) + 2 * k_linear
BIC_linear = n_linear * np.log(RSS_linear / n_linear) + k_linear * np.log(n_linear)

print("Linear Model:")
print("F-stats:", F_linear)
print("AIC:", AIC_linear)
print("BIC:", BIC_linear)

# Parabolik Model
RSS_parabolic = np.sum(residuals_parabolic**2)
RSS_parabolic_full = np.sum((OC2 - np.mean(OC2))**2)
n_parabolic = len(T2)
k_parabolic = 3  # Parabolik modeldeki serbestlik derecesi

F_parabolic = ((RSS_parabolic_full - RSS_parabolic) / (k_parabolic - 0)) / (RSS_parabolic / (n_parabolic - k_parabolic))
AIC_parabolic = n_parabolic * np.log(RSS_parabolic / n_parabolic) + 2 * k_parabolic
BIC_parabolic = n_parabolic * np.log(RSS_parabolic / n_parabolic) + k_parabolic * np.log(n_parabolic)

print("Parabolic Model:")
print("F-stats:", F_parabolic)
print("AIC:", AIC_parabolic)
print("BIC:", BIC_parabolic)

# Çevrimsel Model
RSS_cevrimsel = np.sum(residuals_cevrimsel**2)
RSS_cevrimsel_full = np.sum((OC2 - np.mean(OC2))**2)
n_cevrimsel = len(T2)
k_cevrimsel = 2  # Çevrimsel modeldeki serbestlik derecesi

F_cevrimsel = ((RSS_cevrimsel_full - RSS_cevrimsel) / (k_cevrimsel - 0)) / (RSS_cevrimsel / (n_cevrimsel - k_cevrimsel))
AIC_cevrimsel = n_cevrimsel * np.log(RSS_cevrimsel / n_cevrimsel) + 2 * k_cevrimsel
BIC_cevrimsel = n_cevrimsel * np.log(RSS_cevrimsel / n_cevrimsel) + k_cevrimsel * np.log(n_cevrimsel)

print("Cyclic Model:")
print("F-stats:", F_cevrimsel)
print("AIC:", AIC_cevrimsel)
print("BIC:", BIC_cevrimsel)

# Lineer + Çevrimsel Model
RSS_lc = np.sum(residuals_lc**2)
RSS_lc_full = np.sum((OC2 - np.mean(OC2))**2)
n_lc = len(T2)
k_lc = 4  #Lineer + Çevrimsel modeldeki serbestlik derecesi

F_lc = ((RSS_lc_full - RSS_lc) / (k_lc - 0)) / (RSS_lc / (n_lc - k_lc))
AIC_lc = n_lc * np.log(RSS_lc / n_lc) + 2 * k_lc
BIC_lc = n_lc * np.log(RSS_lc / n_lc) + k_lc * np.log(n_lc)

print("Linear + Cyclic Model:")
print("F-stats:", F_lc)
print("AIC:", AIC_lc)
print("BIC:", BIC_lc)

#Parabolik + Çevrimsel Model
RSS_pc = np.sum(residuals_pc**2)
RSS_pc_full = np.sum((OC2 - np.mean(OC2))**2)
n_pc = len(T2)
k_pc = 5  #Parabolik + Çevrimsel modeldeki serbestlik derecesi

F_pc = ((RSS_pc_full - RSS_pc) / (k_pc - 0)) / (RSS_pc / (n_pc - k_pc))
AIC_pc = n_pc * np.log(RSS_pc / n_pc) + 2 * k_pc
BIC_pc = n_pc * np.log(RSS_pc / n_pc) + k_pc * np.log(n_pc)

print("Parabolic + Cyclic Model:")
print("F-stats:", F_pc)
print("AIC:", AIC_pc)
print("BIC:", BIC_pc)

#En iyi modelin belirlenmesi
F_values = [F_linear, F_parabolic, F_cevrimsel, F_lc, F_pc]
AIC_values = [AIC_linear, AIC_parabolic, AIC_cevrimsel, AIC_lc, AIC_pc]
BIC_values = [BIC_linear, BIC_parabolic, BIC_cevrimsel, BIC_lc, BIC_pc]

best_model_index = np.argmin(AIC_values)  #En düşük AIC hesaplama

best_model = ["Linear", "Parabolic", "Cyclic", "Linear + Cyclic", "Parabolic + Cyclic"][best_model_index]
best_residuals = [residuals_linear, residuals_parabolic, residuals_cevrimsel, residuals_lc, residuals_pc][best_model_index]
best_errors = errors


plt.figure(figsize=(10, 6))
plt.errorbar(T2, OC2, yerr=OC_err2, fmt='o', label='Data')
plt.plot(T2, [result_linear.best_fit, result_parabolic.best_fit, result_cevrimsel.best_fit, result_lc.best_fit, result_pc.best_fit][best_model_index], 'r-', label='Best Fit: ' + best_model)
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C')
plt.title("O-C with Best Fit Model")
plt.show()


plt.figure(figsize=(10, 6))
plt.errorbar(T2, best_residuals, yerr=best_errors, fmt='o', label='Residuals')
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('O-C$_{Residuals}$')
plt.title("Residuals with Error Bars for Best Fit Model")
plt.show()