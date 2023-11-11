load noise.mat
fs = 2000;
%PSD1(noise,fs)
%function PSD1(x,fs)
%window=rectwin(length(x));
%periodogram(x,window,length(x),fs); %直接法 
%end
%% periodogram
figure;
% periodogram 默认使用 Hamming 窗口，所以这里必须指定窗口类型为矩形窗，否则两种方式计算功率谱密度不完全一样
periodogram(noise, rectwin(length(noise)), length(noise), fs);
%% FFT
figure;
N = length(noise)/10;
Y = fft(noise);
Y = Y(1:N/2+1);
xpsd = abs(Y).^2/(N*fs);
xpsd(2:end-1) = 2*xpsd(2:end-1);
f = 0:fs/N:fs/2;
plot(f, 10*log10(xpsd));
grid on;
xlabel('Frequency(Hz)');
ylabel('Power/Frequency(db/Hz)');
title('Periodogram Using FFT');
%% 间接法
n=1:1/fs:1;
nfft=fs/10;
cxn=xcorr(noise,'unbiased'); %计算序列的自相关函数
CXk=fft(cxn,nfft);
Pxx=2*abs(CXk)/fs;
index=0:round(nfft/2-1);
k=index*fs/nfft;
plot_Pxx=10*log10(Pxx(index+1));
figure;
plot(k,plot_Pxx);
grid on;
xlabel('Frequency(Hz)');
ylabel('Power/Frequency(db/Hz)');
title('Periodogram Using auto correlation');