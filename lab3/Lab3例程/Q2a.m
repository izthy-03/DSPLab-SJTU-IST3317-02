fs=1000;
f1=50;
f2=55;
N=2000;
Nfft=N;
n=1:N;
t=n/fs;%采用的时间序列
f=linspace(-fs/2,fs/2-1,N);%频域横坐标，注意奈奎斯特采样定理，最大原信号最大频率不超过采样频率的一半
y = 2*sin(2*pi*f1*t)+3*sin(2*pi*f2*t)+1*randn(1,N)+0.5;
figure;
plot(n,y);
xlabel('n');
ylabel('y');
title('y in time domain');
Y = fftshift(fft(y/N));%用fft得出离散傅里叶变换
figure;
plot(f,abs(Y));%画双侧频谱幅度图
xlabel("f/Hz")
ylabel("Y")
title('Y in fre domain');
grid on
%%
figure;
periodogram(y, hamming(2000), length(y), fs);
%%
figure;
pwelch(y);