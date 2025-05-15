%% 2 s Simultaneous 10-Tone Multitone (no bursts, no Laplace/pink)

Fs       = 48000;    % sampling rate
duration = 2;        % total length (s)
numTones = 10;       % how many simultaneous tones
rng(41,'twister');

% build time vector
N = duration*Fs;
t = (0:N-1)/Fs;

% define log-spaced bands
f_min   = 60;
f_max   = 3000;
edges   = logspace(log10(f_min), log10(f_max), numTones+1);

% pick one random freq in each band
toneFreqs = zeros(1,numTones);
for i = 1:numTones
  toneFreqs(i) = edges(i) + (edges(i+1)-edges(i))*rand();
end

% generate and sum the tones
x = zeros(1,N);
for i = 1:numTones
  phi = 2*pi * rand();  % random phase [0,2π)
  x = x + sin(2*pi*toneFreqs(i)*t + phi);
end

% normalize to ±0.99
peakVal = max(abs(x));
if peakVal > 0
    x = x/peakVal * 0.99;
end

% play & save
sound(x,Fs);
audiowrite('multitone2s.wav', x, Fs, 'BitsPerSample', 32);

% (optional) display the chosen freqs
disp('10 random tones (Hz):');
disp(toneFreqs);


%% Calculate crest factor
peakVal = max(abs(x));
rmsVal  = sqrt(mean(x.^2));
crestFactor = peakVal / rmsVal;

disp(['Crest Factor: ', num2str(crestFactor)]);

%% Plot logarithmic spectrogram
figure;
window  = 1024;
overlap = window / 2;
nfft    = 2^nextpow2(window);
[S, f, t_spec] = spectrogram(x, window, overlap, nfft, Fs);

f(f == 0) = min(f(f > 0));

S_dB = 20*log10(abs(S));

h = pcolor(t_spec, f, S_dB);
set(h, 'EdgeColor', 'none');
shading flat;
set(gca, 'YScale', 'log');
set(gca, 'YDir', 'normal');

yticks([50, 100, 500, 1000, 5000]);
yticklabels({'50Hz', '100Hz', '500Hz', '1kHz', '5kHz'});

clim([0 40]);
colormap(parula);
colorbar;

title('Logarithmic Spectrogram of Generated Signal');
xlabel('Time (s)');
ylabel('Frequency (log scale)');


%% Create Amplitude distribution
edges = linspace(-1, 1, 101);

counts = histcounts(x, edges);

countsPerc = (counts / sum(counts)) * 100;

binCenters = edges(1:end-1) + diff(edges)/2;

figure;
bar(binCenters, countsPerc, 'hist');
xlim([-1, 1]);
xlabel('Amplitude');
ylabel('Percentage of Samples (%)');
title('Amplitude Distribution');

%% Create Frequency amplitude graph
N = length(x);
X = fft(x);
X = X(1:floor(N/2));
f = (0:floor(N/2)-1) * (Fs/N);

idx = f >= 20;
f_lin = f(idx);
dB = 20*log10(abs(X(idx)));

f_target = logspace(log10(60), log10(3000), 1000);
dB_interp = interp1(f_lin, dB, f_target, 'linear');

smoothed_dB = smoothdata(dB_interp, 'movmean', 50);

figure;
semilogx(f_target, smoothed_dB, 'b-', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
title('Frequency amplitude spectrum');
grid on;

xticks([50 100 200 500 1000 2000 5000]);
xticklabels({'50 Hz','100 Hz','200 Hz','500 Hz','1 kHz','2 kHz','5 kHz'});

%% Wavefile
t = (0 : length(x)-1) / Fs;
figure;
plot(t, x, 'k');             % 'k' = black line
xlabel('Time [s]');
ylabel('Amplitude');
title('Time-Domain Waveform of Shaped Multitone Signal');
grid on;





