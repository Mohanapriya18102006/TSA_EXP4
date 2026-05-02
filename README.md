# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# Date: 02-05-2026



### AIM:

To implement ARMA model in python.

### ALGORITHM:

1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.
6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_excel('/content/temperature_data.xlsx')

# Set sample size
N = 1000

# Set figure size
plt.rcParams['figure.figsize'] = [12, 6]

# Convert 'Date' column to datetime and set as index, then drop NaN values from Temperature
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data = data.dropna(subset=['Date']) # Drop rows where 'Date' could not be parsed
data = data.set_index('Date')
data = data.sort_index() # Sort the DataFrame by its datetime index to ensure monotonicity

# Extract data (using 'Temperature' column)
X = data['Temperature'].dropna() # Drop NaN values from the Temperature series

# Plot original data
plt.plot(X)
plt.title('Original Data')
plt.show()

# Plot ACF and PACF of original data
plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# =========================
# ARMA(1,1) Model
# =========================

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

# Plot ARMA(1,1)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# ACF & PACF
plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()


# ARMA(2,2) Model


arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)

# Plot ARMA(2,2)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# ACF & PACF
plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```


OUTPUT:

Original data:

<img width="986" height="520" alt="image" src="https://github.com/user-attachments/assets/3d6f2eb9-fd73-4d59-be46-9d84c094ba7f" />

Autocorrelation:

<img width="983" height="242" alt="image" src="https://github.com/user-attachments/assets/26dde716-eabc-41c1-8407-a765de84da9f" />

Partial Autocorrelation:

<img width="982" height="238" alt="image" src="https://github.com/user-attachments/assets/e4a9cc41-66fd-4c8b-ac02-ee8acbf93e2f" />

Stimulated ARMA(1,1) Process:

<img width="977" height="516" alt="image" src="https://github.com/user-attachments/assets/5b6283f8-b1d9-48ab-8202-7d815d03de57" />

Autocorrelation:

<img width="972" height="517" alt="image" src="https://github.com/user-attachments/assets/a3add9cc-d079-47f0-afef-1add71146e59" />

Patial Autocorrelation:

<img width="975" height="518" alt="image" src="https://github.com/user-attachments/assets/f5d9f99c-f1b8-45dd-9966-84ba916a37f3" />

Stimulated ARMA(2,2) Process:

<img width="971" height="519" alt="image" src="https://github.com/user-attachments/assets/63de392d-8956-4bc0-9592-23ad72dad945" />

Autocorrelation:

<img width="975" height="523" alt="image" src="https://github.com/user-attachments/assets/b54b0edc-858b-4067-8416-a6d64dc04836" />

Partial Autocorrelation:

<img width="980" height="508" alt="image" src="https://github.com/user-attachments/assets/2ba379af-4a1f-4590-b135-6d10bbfc3d67" />










RESULT:

Thus, a python program is created to fit ARMA Model successfully.
