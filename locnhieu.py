# Rest of your code
import numpy as np
import pandas as pd
from PyEMD import EEMD
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Đọc dữ liệu từ file CSV
data = pd.read_csv("/content/Loivongtrong.csv", header=None)
signal = data.iloc[:, 0].values  # Lấy cột đầu tiên làm tín hiệu mẫu
n = len(signal)  # Số điểm dữ liệu: 1000
total_time = 10  # Thời gian: 10 giây
t = np.linspace(0, total_time, n)  # Trục thời gian
sample_rate = n / total_time  # Tần số lấy mẫu: 100 Hz

# Khởi tạo EEMD (mô phỏng AREEMD)
eemd = EEMD()
eemd.noise_seed(12345)  # Đặt seed để tái lập kết quả
eemd.trials = 50  # Số lần lặp ensemble (giảm để tối ưu thời gian)
eemd.noise_width = 0.2  # Độ lớn nhiễu trắng

# Phân rã tín hiệu thành các IMF
IMFs = eemd.eemd(signal, t)

# Phân tích tần số của các IMF
freqs = fftfreq(n, 1/sample_rate)  # Tần số tương ứng
positive_freqs = freqs[:n//2]  # Chỉ lấy tần số dương (0 đến 50 Hz)

# Lọc nhiễu: Loại bỏ IMF tần số cao (giả định IMF 1, 2 là nhiễu)
filtered_signal = np.sum(IMFs[2:], axis=0)  # Tổng hợp IMF từ 3 trở đi

# Tính SNR để đánh giá
def calculate_snr(original, filtered):
    signal_power = np.mean(original**2)
    noise_power = np.mean((original - filtered)**2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

snr = calculate_snr(signal, filtered_signal)

# Vẽ kết quả
plt.figure(figsize=(12, 10))

# Tín hiệu gốc
plt.subplot(3, 1, 1)
plt.plot(t, signal, 'b', label='Tín hiệu gốc')
plt.title('Tín hiệu gốc')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.legend()

# Tín hiệu đã lọc
plt.subplot(3, 1, 2)
plt.plot(t, filtered_signal, 'r', label='Tín hiệu đã lọc')
plt.title(f'Tín hiệu đã lọc (SNR: {snr:.2f} dB)')
plt.xlabel('Thời gian (giây)')
plt.ylabel('Biên độ')
plt.legend()

# Phổ tần số của các IMF đầu tiên
plt.subplot(3, 1, 3)
for i in range(min(3, len(IMFs))):  # Vẽ phổ tần số của 3 IMF đầu
    fft_imf = np.abs(fft(IMFs[i]))[:n//2]
    plt.plot(positive_freqs, fft_imf, label=f'IMF {i+1}')
plt.title('Phổ tần số của các IMF đầu tiên')
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ')
plt.legend()
plt.xlim(0, 50)  # Giới hạn trục tần số từ 0 đến 50 Hz (Nyquist frequency)

plt.tight_layout()
plt.show()

# In thông tin
print(f"Số lượng IMF: {len(IMFs)}")
print(f"Tần số lấy mẫu: {sample_rate:.2f} Hz")
print(f"SNR sau khi lọc: {snr:.2f} dB")