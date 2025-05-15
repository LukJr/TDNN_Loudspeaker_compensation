%% Parameters
Fs       = 48000;         % Sampling rate
duration = 10;            % Duration in seconds
N        = Fs * duration; % Total samples
b        = 1;           % Laplace scale parameter
rng(42,'twister');

%% 1) Gaussian white noise → pink noise via 1/f FFT filter
w = randn(N,1);
W = fft(w);
f = (0:N-1)' * (Fs/N);
scaling = ones(N,1);
scaling(2:floor(N/2)+1) = 1 ./ sqrt(f(2:floor(N/2)+1));
if mod(N,2)==0
    scaling(floor(N/2)+2:end) = 1 ./ sqrt(f(floor(N/2):-1:2));
else
    scaling(floor(N/2)+2:end) = 1 ./ sqrt(f(floor(N/2)+1:-1:2));
end
P = real(ifft(W .* scaling));    % Pink-1/f noise (still Gaussian marginal)

%% 2) Compute the sample‐std of P (for CDF)
sigmaP = std(P);

%% 3) Map P→Uniform via Gaussian CDF
u = 0.5 * (1 + erf(P./(sqrt(2)*sigmaP)));  

%% 4) Inverse‐Laplace CDF on u to get Laplace marginal
L = zeros(N,1);
mask = u < 0.5;
L(mask)   =  b * log(2*u(mask));
L(~mask)  = -b * log(2*(1 - u(~mask)));

%% 5) Normalize to [-1,1]
x = L / max(abs(L));

%% 6) Playback & save
sound(x, Fs);
audiowrite('laplace_pinknoise.wav', x, Fs, 'BitsPerSample',32);

%% 7) Crest factor
crest = max(abs(x)) / sqrt(mean(x.^2));
fprintf('Crest Factor: %.2f\n', crest);

%% 8) Log Spectrogram
figure;
window  = 1024; overlap = 512; nfft = 2^nextpow2(window);
[S,f_s,t_s] = spectrogram(x, window, overlap, nfft, Fs);
f_s(f_s==0) = min(f_s(f_s>0));
imagesc(t_s, log10(f_s), 20*log10(abs(S)));
axis xy; colormap parula; colorbar;
yticks(log10([20 50 100 500 1e3 5e3 1e4 2e4]));
yticklabels({'20','50','100','500','1k','5k','10k','20k'});
title('Log‐Freq Spectrogram'); xlabel('Time (s)'); ylabel('log10(Freq)');

%% 9) Amplitude Distribution
edges = linspace(-1,1,101);
cnts  = histcounts(x, edges);
pct   = cnts/sum(cnts)*100;
bins  = edges(1:end-1)+diff(edges)/2;
figure; bar(bins, pct, 'hist');
xlim([-1 1]); xlabel('Amplitude'); ylabel('% Samples');
title('Amplitude Distribution');

%% 10) Frequency‐Amplitude Spectrum
Xf = fft(x); Xf = Xf(1:floor(N/2));
f_lin = (0:floor(N/2)-1)*(Fs/N);
idx   = f_lin>=20;
f_t   = logspace(log10(20),log10(20000),1000);
dB    = 20*log10(abs(Xf(idx)));
dB_i  = interp1(f_lin(idx), dB, f_t);
dB_s  = smoothdata(dB_i,'movmean',50);
figure; semilogx(f_t, dB_s, 'LineWidth',1.5);
grid on; xlabel('Freq (Hz)'); ylabel('Amplitude (dB)');
title('Freq‐Amplitude Spectrum');
xticks([20 50 100 200 500 1e3 2e3 5e3 1e4 2e4]);
xticklabels({'20','50','100','200','500','1k','2k','5k','10k','20k'});
