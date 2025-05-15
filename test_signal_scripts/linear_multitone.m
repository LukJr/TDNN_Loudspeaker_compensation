%% Repetitive Multitone Signal Generation with Full Amplitude Shaping
% This script generates a multitone signal covering 60 Hz to 3 kHz,
% repeats it with multiple random-phase realizations, discards the first 
% period of each realization for settling, and applies amplitude shaping 
% (from Gaussian to a Laplace distribution with frequency-domain re-shaping)
% to every period used for analysis.

rng(43, 'twister'); 

%% 1. Define Sampling and Frequency Parameters
fs = 48000;             % Sampling frequency (Hz)

% Set the period to be that of the lowest excited frequency (60 Hz)
f_min = 60;             % Minimum frequency (Hz)
T = 0.25;        % Desired period of 0.25 seconds
N = fs * T;      

% Frequency band of interest
f_max = 3000;           % Maximum frequency (Hz)

% Fundamental frequency (frequency resolution)
f0 = 1/T;            

% Determine the harmonic indices corresponding to the desired frequency range
k_min = ceil(f_min / f0);
k_max = floor(f_max / f0); 

% Create a time vector for one period
t = (0:N-1) / fs;

%% 2. Generate the Repetitive Multitone Sequence
M = 5;    % Number of different random-phase realizations
P = 6;    % Total periods per realization (the first period will be discarded during analysis)

% Preallocate the full sequence array
multitone_seq = [];

for m = 1:M
    % Generate one multitone period for this realization
    u_m = zeros(1, N);
    for k = k_min:k_max
        phase = 2*pi*rand;   % Random phase in [0, 2π)
        u_m = u_m + cos(2*pi*k*f0*t + phase);
    end
    disp(['before ']);
    disp(['Min: ', num2str(min(u_m))]);
    disp(['Max: ', num2str(max(u_m))]);
    disp(['after ']);
    u_m = u_m - mean(u_m);   % Remove any DC component.
    u_m = u_m / std(u_m);      % Scale the variance to 1.
    %u_m = u_m / max(abs(u_m));  % Normalize the period
    disp(['Min: ', num2str(min(u_m))]);
    disp(['Max: ', num2str(max(u_m))]);
    % Repeat this realization for (P+1) periods.
    % The first period of each realization is for settling and will be excluded from analysis.
    multitone_seq = [multitone_seq, repmat(u_m, 1, (P+1))];
end

%% 3. Apply Amplitude Shaping with Frequency-Domain Reshaping
% Create a new sequence where all periods (or, if desired, 
% only those used for analysis) are amplitude shaped.

% Copy the original sequence for shaping
multitone_seq_shaped = multitone_seq;

%multitone_time_shape = multitone_seq;

% Total number of periods in the sequence
totalPeriods = M * (P+1);

for idx = 1:totalPeriods
    % Optionally skip the settling period:
    % Uncomment the following block if you want to skip shaping for settling periods.
    %{
    if mod(idx-1, P+1) == 0
        continue; % Skip the first period of each realization.
    end
    %}
    
    % Determine the starting and ending sample indices for this period
    period_start = (idx-1)*N + 1;
    period_end = idx*N;
    
    % Extract the current period from the sequence
    current_period = multitone_seq(period_start:period_end);
    
    % --- Amplitude Shaping Transformation (Time Domain) ---
    % Convert the signal's Gaussian-like distribution to a uniform distribution in [-0.5, 0.5]
    u_uniform = (1 + erf(current_period / sqrt(2)))/2 - 0.5;
    
    % Convert the uniform distribution to a Laplace distribution:
    % y = -sign(u) * log(1 - 2|u|)
    current_period_shaped = -sign(u_uniform) .* log(1 - 2*abs(u_uniform));

    multitone_time_shape(period_start:period_end) = current_period_shaped;
    
    % --- Frequency-Domain Reshaping ---
    % Compute the FFT of the Laplace-shaped period.
    Y = fft(current_period_shaped);
    
    % Construct the target amplitude spectrum A.
    % Here, we define A=1 for the frequency bins corresponding to the excited band and 0 elsewhere.
    A = zeros(size(Y));
    % MATLAB FFT: Index 1 is DC; indices 2 to (N/2+1) are positive frequencies.
    A(k_min+1 : k_max+1) = 1;  % For positive frequencies.
    
    % For real signals, mirror the spectrum for negative frequencies.
    A(N - k_max + 1 : N - k_min + 1) = 1;
    
    % Avoid division by zero: for bins where |Y| == 0, set the magnitude to 1 temporarily.
    magY = abs(Y);
    magY(magY == 0) = 1;
    
    % Impose the target amplitude while preserving the phase.
    Y_reshaped = A .* (Y ./ magY);
    
    % Transform back to the time domain.
    current_period_reshaped = real(ifft(Y_reshaped));
    
    % Replace the original period with the reshaped period.
    multitone_seq_shaped(period_start:period_end) = current_period_reshaped;
end

% Use the amplitude-shaped sequence for playback and saving
x = multitone_seq_shaped;

%% Signal normalization
peakVal = max(abs(x));
if peakVal > 0
    x = x / peakVal;
end

%peakVal = max(abs(multitone_time_shape));
%if peakVal > 0
%    multitone_time_shape = multitone_time_shape / peakVal;
%end

%% 4. Play and Save the Signal
sound(x, fs);
audiowrite('repetitive_multitone_shaped.wav', x, fs, 'BitsPerSample', 32);


%% 5. Calculate Crest Factor
peakVal = max(abs(x));
rmsVal  = sqrt(mean(x.^2));
crestFactor = peakVal / rmsVal;
disp(['Crest Factor: ', num2str(crestFactor)]);

%% 5. Calculate Crest Factor for only time shaape
peakVal2 = max(abs(multitone_time_shape));
rmsVal2  = sqrt(mean(multitone_time_shape.^2));
crestFactor2 = peakVal2 / rmsVal2;
disp(['Crest Factor for time shape: ', num2str(crestFactor2)]);

%% 6. Plot Logarithmic Spectrogram
figure;
window  = 1024;
overlap = window / 2;
nfft    = 2^nextpow2(window);
[S, f_spec, t_spec] = spectrogram(x, window, overlap, nfft, fs);

% Avoid log(0) issues
f_spec(f_spec == 0) = min(f_spec(f_spec > 0));

S_dB = 20*log10(abs(S));

h = pcolor(t_spec, f_spec, S_dB);
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

%% 7. Create Amplitude Distribution Plot
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


%% 8. Create Frequency Amplitude Graph
N_total = length(x);
X = fft(x);
X = X(1:floor(N_total/2));
f_lin = (0:floor(N_total/2)-1) * (fs/N_total);

% Only consider frequencies above 20 Hz
idx = f_lin >= 20;
f_lin = f_lin(idx);
dB = 20*log10(abs(X(idx)));

f_target = logspace(log10(60), log10(3000), 1000);
dB_interp = interp1(f_lin, dB, f_target, 'linear');

smoothed_dB = smoothdata(dB_interp, 'movmean', 50);

figure;
semilogx(f_target, smoothed_dB, 'b-', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
title('Frequency Amplitude Spectrum');
grid on;
xticks([50 100 200 500 1000 2000 5000]);
xticklabels({'50 Hz','100 Hz','200 Hz','500 Hz','1 kHz','2 kHz','5 kHz'});


%% Plot the signal in the time domain
t = (0 : length(x)-1) / fs;
figure;
plot(t, x, 'k');             % 'k' = black line
xlabel('Time [s]');
ylabel('Amplitude');
title('Time-Domain Waveform of Shaped Multitone Signal');
grid on;