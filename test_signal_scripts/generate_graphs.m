%% Load WAV file
[raw, Fs] = audioread('true_laplace_pinknoise.FFT.Filtered.wav');
% stereo to mono if needed:
if size(raw,2)>1
    x = mean(raw,2);
else
    x = raw;
end
% normalize
x = x / max(abs(x));

%% Crest factor
crest = max(abs(x)) / sqrt(mean(x.^2));
fprintf('Crest Factor: %.2f\n', crest);

%% Log-Frequency Spectrogram
figure;
window  = 1024;
overlap = window/2;
nfft    = 2^nextpow2(window);
[S,f_s,t_s] = spectrogram(x, window, overlap, nfft, Fs);
f_s(f_s==0) = min(f_s(f_s>0));           % avoid log(0)
imagesc(t_s, log10(f_s), 20*log10(abs(S)));
axis xy;
colormap parula;
colorbar;
yticks(log10([20 50 100 500 1e3 5e3 1e4 2e4]));
yticklabels({'20','50','100','500','1k','5k','10k','20k'});
title('Log‐Freq Spectrogram');
xlabel('Time (s)');
ylabel('log_{10}(Frequency)');

%% Amplitude Distribution Histogram
edges = linspace(-1,1,101);
cnts  = histcounts(x, edges);
pct   = cnts/sum(cnts)*100;
bins  = edges(1:end-1) + diff(edges)/2;
figure;
bar(bins, pct, 'hist');
xlim([-1 1]);
xlabel('Amplitude');
ylabel('% of Samples');
title('Amplitude Distribution');

%% Frequency‐Amplitude Spectrum
N    = length(x);
Xf   = fft(x);
Xf   = Xf(1:floor(N/2));
f_lin= (0:floor(N/2)-1)*(Fs/N);
idx  = f_lin >= 20;
f_t  = logspace(log10(20), log10(20000), 1000);
dB   = 20*log10(abs(Xf(idx)));
dB_i = interp1(f_lin(idx), dB, f_t, 'linear');
dB_s = smoothdata(dB_i, 'movmean', 50);

figure;
semilogx(f_t, dB_s, 'LineWidth',1.5);
grid on;
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
title('Frequency–Amplitude Spectrum');
xticks([20 50 100 200 500 1e3 2e3 5e3 1e4 2e4]);
xticklabels({'20','50','100','200','500','1k','2k','5k','10k','20k'});

%% Plot the signal in the time domain
t = (0 : length(x)-1) / fs;
figure;
plot(t, x, 'k');             % 'k' = black line
xlabel('Time [s]');
ylabel('Amplitude');
title('Time-Domain Waveform of Shaped Multitone Signal');
grid on;