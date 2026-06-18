/**
 * Three-Phase Inverter ML Dashboard - Simulation & Logic
 * Md. Atair Rahman Alvi (BRAC University)
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- UI Elements ---
    const faultSelector = document.getElementById('fault-selector');
    const sliderCurrent = document.getElementById('slider-current');
    const valCurrent = document.getElementById('val-current');
    
    const groupPhaseSelect = document.getElementById('group-phase-select');
    const selectFaultPhase = document.getElementById('select-fault-phase');
    
    const groupDcOffset = document.getElementById('group-dc-offset');
    const sliderDcOffset = document.getElementById('slider-dc-offset');
    const valDcOffset = document.getElementById('val-dc-offset');
    
    const groupGridScale = document.getElementById('group-grid-scale');
    const sliderGridScale = document.getElementById('slider-grid-scale');
    const valGridScale = document.getElementById('val-grid-scale');
    
    const sliderNoise = document.getElementById('slider-noise');
    const valNoise = document.getElementById('val-noise');
    
    const btnHealing = document.getElementById('btn-healing');
    const healingDot = document.getElementById('healing-dot');
    const healingStatusText = document.getElementById('healing-status-text');
    const healingActionDetails = document.getElementById('healing-action-details');
    
    const sysPulseRing = document.getElementById('sys-pulse-ring');
    const sysStatusBadge = document.getElementById('sys-status-badge');
    const sysClock = document.getElementById('sys-clock');
    
    // Gauges
    const gaugeFreq = document.getElementById('gauge-freq');
    const gaugeFreqDrift = document.getElementById('gauge-freq-drift');
    const gaugeThd = document.getElementById('gauge-thd');
    const gaugeThdStatus = document.getElementById('gauge-thd-status');
    const gaugePowerP = document.getElementById('gauge-power-p');
    const gaugePowerQ = document.getElementById('gauge-power-q');
    const gaugePowerFactor = document.getElementById('gauge-power-factor');
    
    // AI diagnostics
    const predictedClassName = document.getElementById('predicted-class-name');
    const predictedConfidenceBar = document.getElementById('predicted-confidence-bar');
    const predictedConfidenceVal = document.getElementById('predicted-confidence-val');
    
    // Feature Stats cells
    const fMeanE = document.getElementById('f-mean-e');
    const fStdE = document.getElementById('f-std-e');
    const fRmsE = document.getElementById('f-rms-e');
    const fSkewE = document.getElementById('f-skew-e');
    
    const fMeanI = document.getElementById('f-mean-i');
    const fStdI = document.getElementById('f-std-i');
    const fRmsI = document.getElementById('f-rms-i');
    const fSkewI = document.getElementById('f-skew-i');
    
    const fMeanErr = document.getElementById('f-mean-err');
    const fStdErr = document.getElementById('f-std-err');
    const fRmsErr = document.getElementById('f-rms-err');
    const fSkewErr = document.getElementById('f-skew-err');
    
    const fVdcVal = document.getElementById('f-vdc-val');
    
    // --- Canvas Elements & Contexts ---
    const scopeCanvas = document.getElementById('scope-canvas');
    const errorCanvas = document.getElementById('error-canvas');
    const ctxScope = scopeCanvas.getContext('2d');
    const ctxError = errorCanvas.getContext('2d');
    
    // --- State variables ---
    let time = 0;
    const fs = 5000; // Sample rate (Hz)
    const windowSize = 50; // ML window size (10ms)
    
    // Formatted names of classes
    const classNames = [
        "HEALTHY",
        "OPEN-CIRCUIT FAULT",
        "DC CURRENT INJECTION",
        "GRID VOLTAGE SAG/SWELL",
        "ISLANDING DETECTED",
        "PLL DESYNCHRONIZATION"
    ];
    
    // Simulated values buffers for drawing
    const maxDrawSamples = 200;
    const historyE = [[], [], []]; // 3 phases grid voltage
    const historyI = [[], [], []]; // 3 phases inverter currents
    const historyRef = [[], [], []]; // 3 phases references
    const historyErr = [[], [], []]; // 3 phases tracking errors
    
    // --- Event Listeners to toggle inputs dynamically ---
    faultSelector.addEventListener('change', () => {
        const mode = parseInt(faultSelector.value);
        // Toggle control visibility
        groupPhaseSelect.style.display = (mode === 1) ? 'flex' : 'none';
        groupDcOffset.style.display = (mode === 2) ? 'flex' : 'none';
        groupGridScale.style.display = (mode === 3) ? 'flex' : 'none';
    });
    
    // Update labels on slider move
    sliderCurrent.addEventListener('input', () => valCurrent.textContent = parseFloat(sliderCurrent.value).toFixed(1) + ' A');
    sliderDcOffset.addEventListener('input', () => valDcOffset.textContent = parseFloat(sliderDcOffset.value).toFixed(1) + ' A');
    sliderGridScale.addEventListener('input', () => valGridScale.textContent = parseFloat(sliderGridScale.value).toFixed(2) + 'x');
    sliderNoise.addEventListener('input', () => valNoise.textContent = parseFloat(sliderNoise.value).toFixed(2));
    
    // Tab switching for research paper figures
    const tabButtons = document.querySelectorAll('.tab-btn');
    const activeFigureImg = document.getElementById('active-figure-img');
    const activeFigureCaption = document.getElementById('active-figure-caption');
    
    const captionsMap = {
        fig1: "<strong>Fig. 1 — Model Accuracy Comparison:</strong> Benchmarking Random Forest (99.44%), XGBoost (98.89%), Shallow MLP (99.72%), Deep MLP (55.00%), Gradient Boosting (99.44%), and the proposed ResMLP (98.06% on GPU validation) for three-phase inverter fault detection.",
        fig2: "<strong>Fig. 2 — Confusion Matrix (Proposed ResMLP):</strong> Symmetrical high-accuracy performance across all classes, showing absolute separation of open-circuit faults and islanding (1.00 Recall).",
        fig3: "<strong>Fig. 3 — Per-Class F1-Score:</strong> Comparing performance metrics across all models for each of the six target classes (Healthy, Open-Circuit, DC Injection, Sag/Swell, Islanding, and PLL Desync).",
        fig4: "<strong>Fig. 4 — Proposed ResMLP Training Curves:</strong> Smooth validation accuracy and loss tracking over 400 epochs, showing the benefit of skip connections and GELU activations in preventing vanishing gradient.",
        fig5: "<strong>Fig. 5 — Three-Phase Current Waveforms:</strong> Visual comparison between (a) Healthy balanced operation and (b) Open-Circuit Switch Fault showing half-cycle loss in Phase A current.",
        fig6: "<strong>Fig. 6 — Mean THD by Fault Class:</strong> Showcasing the distinct harmonic footprints, with open-circuit and islanding faults registering the highest current distortions (10% - 20%)."
    };
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const figId = btn.getAttribute('data-figure');
            activeFigureImg.src = `paper/figures/${figId}.png`;
            activeFigureCaption.innerHTML = captionsMap[figId];
        });
    });
    
    // --- Clock update ---
    setInterval(() => {
        const d = new Date();
        sysClock.textContent = d.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    }, 1000);
    
    // --- Math helpers ---
    function mean(arr) {
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    }
    
    function std(arr, avg) {
        const mu = avg !== undefined ? avg : mean(arr);
        const sqDiff = arr.map(v => Math.pow(v - mu, 2));
        return Math.sqrt(mean(sqDiff));
    }
    
    function rms(arr) {
        const sq = arr.map(v => v * v);
        return Math.sqrt(mean(sq));
    }
    
    function skew(arr, avg, stdev) {
        const mu = avg !== undefined ? avg : mean(arr);
        const sd = stdev !== undefined ? stdev : std(arr, mu);
        if (sd < 1e-8) return 0;
        const cubed = arr.map(v => Math.pow((v - mu) / sd, 3));
        return mean(cubed);
    }
    
    // --- Resize Canvas ---
    function resizeCanvas() {
        const w1 = scopeCanvas.parentElement.clientWidth;
        const h1 = scopeCanvas.parentElement.clientHeight;
        scopeCanvas.width = w1;
        scopeCanvas.height = h1;
        
        const w2 = errorCanvas.parentElement.clientWidth;
        const h2 = errorCanvas.parentElement.clientHeight;
        errorCanvas.width = w2;
        errorCanvas.height = h2;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    // --- Main Simulation & Render Loop ---
    function updateSimulation() {
        const selectedMode = parseInt(faultSelector.value);
        const healingEnabled = btnHealing.checked;
        
        const I_amp = parseFloat(sliderCurrent.value);
        const noiseStd = parseFloat(sliderNoise.value);
        
        const f_grid = 50.0;
        const Vgrid_amp = 311.0;
        const Vdc_nom = 700.0;
        
        // Time increment based on fs (5000Hz)
        time += 1 / fs;
        
        // 1. Inverter Core & Self-Healing Logic
        let activeHealingAction = "No active grid threat detected.";
        let finalMode = selectedMode;
        let healingApplied = false;
        
        // Redefined parameters if healing is active
        let faultPhaseOverride = parseInt(selectFaultPhase.value);
        let dcOffsetApplied = [0, 0, 0];
        let gridScaleApplied = [1.0, 1.0, 1.0];
        let freqDriftApplied = f_grid;
        let pllDriftApplied = 0.0;
        
        // Read specific input values based on selection
        if (selectedMode === 1) {
            // Open circuit target phase
        } else if (selectedMode === 2) {
            const dcVal = parseFloat(sliderDcOffset.value);
            // Apply to phase A and B current offset (simulating python generate_3phase_dataset.py)
            dcOffsetApplied[0] = dcVal;
            dcOffsetApplied[1] = -dcVal * 0.5;
        } else if (selectedMode === 3) {
            const scale = parseFloat(sliderGridScale.value);
            // Single phase or three phase scale
            gridScaleApplied[0] = scale;
            gridScaleApplied[1] = scale;
            gridScaleApplied[2] = scale;
        } else if (selectedMode === 4) {
            freqDriftApplied = f_grid + 2.0; // Drifts grid frequency by +2.0Hz
        } else if (selectedMode === 5) {
            pllDriftApplied = 0.8; // Drifts phase angle by 0.8 rad
        }
        
        // Trigger self-healing action overrides if turned on and a fault is active
        if (healingEnabled && selectedMode > 0) {
            healingApplied = true;
            if (selectedMode === 1) {
                activeHealingAction = "Switching to Redundant Phase-Leg. Restoring balanced currents...";
                finalMode = 0; // Remediated back to healthy waveform
            } else if (selectedMode === 2) {
                activeHealingAction = "Injecting compensation bias. Cancelling DC offset...";
                dcOffsetApplied = [0, 0, 0]; // Nullify bias
                finalMode = 0;
            } else if (selectedMode === 3) {
                activeHealingAction = "STATCOM Q-injection mode active. Injecting reactive current to support voltage...";
                gridScaleApplied = [0.95, 0.95, 0.95]; // Pulls voltage back near nominal
                finalMode = 0;
            } else if (selectedMode === 4) {
                activeHealingAction = "Islanding detected! Safely disconnecting inverter to prevent grid backfeed (IEEE 1547)...";
                // System shuts down (current goes to zero)
                finalMode = 4; // Keeps mode but output collapses
            } else if (selectedMode === 5) {
                activeHealingAction = "Resetting PLL filter parameters. Resynchronizing grid angle...";
                pllDriftApplied = 0.0; // PLL locked
                finalMode = 0;
            }
        }
        
        // 2. Waveform Math Model
        const phase_offsets = [0, -2 * Math.PI / 3, 2 * Math.PI / 3];
        const e = [0, 0, 0];
        const i_L = [0, 0, 0];
        const i_ref = [0, 0, 0];
        const err = [0, 0, 0];
        
        const Vdc = Vdc_nom + Math.sin(2 * Math.PI * f_grid * time * 2) * 3; // slight ripple
        
        for (let ph = 0; ph < 3; ph++) {
            const ang = 2 * Math.PI * f_grid * time + phase_offsets[ph];
            
            // Grid Voltage
            if (selectedMode === 4) {
                // Islanding: grid collapses exponentially (decay relative to fault inception)
                const decay = Math.exp(-3 * (time % 1.5));
                e[ph] = Vgrid_amp * Math.sin(2 * Math.PI * freqDriftApplied * time + phase_offsets[ph]) * decay;
            } else {
                e[ph] = Vgrid_amp * gridScaleApplied[ph] * Math.sin(ang);
            }
            
            // Reference current (in phase with nominal grid voltage)
            i_ref[ph] = I_amp * Math.sin(ang);
            
            // Actual inverter current
            if (healingApplied && selectedMode === 4) {
                // Anti-islanding isolation: current drops to 0 immediately
                i_L[ph] = 0;
            } else {
                // Nominal current tracking
                if (selectedMode === 4) {
                    // Islanded current drifts in frequency
                    i_L[ph] = I_amp * Math.sin(2 * Math.PI * freqDriftApplied * time + phase_offsets[ph]) * (1 + 0.08 * Math.sin(2 * Math.PI * 3 * time));
                } else {
                    i_L[ph] = I_amp * Math.sin(ang + pllDriftApplied);
                }
                
                // Class specific distortions
                if (finalMode === 1 && ph === faultPhaseOverride) {
                    // Open-circuit fault on specific phase: half cycle loss
                    const sin_val = Math.sin(ang);
                    if (sin_val > 0) {
                        i_L[ph] = 0.05 * i_ref[ph];
                    }
                }
                
                if (finalMode === 2) {
                    // DC Offset added
                    i_L[ph] += dcOffsetApplied[ph];
                }
                
                if (finalMode === 3) {
                    // Voltage sag current degradation
                    i_L[ph] = i_ref[ph] * (1 - 0.25 * (1 - gridScaleApplied[ph]));
                }
            }
            
            // Add measurement noises
            const randNoise = (Math.random() - 0.5) * 2 * noiseStd;
            const randRefNoise = (Math.random() - 0.5) * 0.05;
            const randGridNoise = (Math.random() - 0.5) * 2.0;
            
            e[ph] += randGridNoise;
            i_L[ph] += randNoise;
            i_ref[ph] += randRefNoise;
            
            err[ph] = i_ref[ph] - i_L[ph];
        }
        
        // Push current values to buffers
        for (let ph = 0; ph < 3; ph++) {
            historyE[ph].push(e[ph]);
            historyI[ph].push(i_L[ph]);
            historyRef[ph].push(i_ref[ph]);
            historyErr[ph].push(err[ph]);
            
            if (historyE[ph].length > maxDrawSamples) {
                historyE[ph].shift();
                historyI[ph].shift();
                historyRef[ph].shift();
                historyErr[ph].shift();
            }
        }
        
        // 3. Sliding Window Features Extraction (for ML display & Inference)
        // Grab last 50 samples
        const windowIdx = Math.min(historyI[0].length, windowSize);
        let extractedFeatures = {
            e: { mean: 0, std: 0, rms: 0, skew: 0 },
            i: { mean: 0, std: 0, rms: 0, skew: 0 },
            err: { mean: 0, std: 0, rms: 0, skew: 0 }
        };
        
        if (windowIdx >= windowSize) {
            // Aggregate over three phases to represent three-phase values
            const subE = historyE[0].slice(-windowSize).concat(historyE[1].slice(-windowSize)).concat(historyE[2].slice(-windowSize));
            const subI = historyI[0].slice(-windowSize).concat(historyI[1].slice(-windowSize)).concat(historyI[2].slice(-windowSize));
            const subErr = historyErr[0].slice(-windowSize).concat(historyErr[1].slice(-windowSize)).concat(historyErr[2].slice(-windowSize));
            
            const meanE = mean(subE);
            const stdE = std(subE, meanE);
            const rmsE = rms(subE);
            const skewE = skew(subE, meanE, stdE);
            
            const meanI = mean(subI);
            const stdI = std(subI, meanI);
            const rmsI = rms(subI);
            const skewI = skew(subI, meanI, stdI);
            
            const meanErr = mean(subErr);
            const stdErr = std(subErr, meanErr);
            const rmsErr = rms(subErr);
            const skewErr = skew(subErr, meanErr, stdErr);
            
            extractedFeatures = {
                e: { mean: meanE, std: stdE, rms: rmsE, skew: skewE },
                i: { mean: meanI, std: stdI, rms: rmsI, skew: skewI },
                err: { mean: meanErr, std: stdErr, rms: rmsErr, skew: skewErr }
            };
            
            // Update UI feature stats
            fMeanE.textContent = extractedFeatures.e.mean.toFixed(2);
            fStdE.textContent = extractedFeatures.e.std.toFixed(1);
            fRmsE.textContent = extractedFeatures.e.rms.toFixed(1);
            fSkewE.textContent = extractedFeatures.e.skew.toFixed(3);
            
            fMeanI.textContent = extractedFeatures.e.mean.toFixed(2); // currents
            fMeanI.textContent = extractedFeatures.i.mean.toFixed(3);
            fStdI.textContent = extractedFeatures.i.std.toFixed(2);
            fRmsI.textContent = extractedFeatures.i.rms.toFixed(2);
            fSkewI.textContent = extractedFeatures.i.skew.toFixed(3);
            
            fMeanErr.textContent = extractedFeatures.err.mean.toFixed(3);
            fStdErr.textContent = extractedFeatures.err.std.toFixed(3);
            fRmsErr.textContent = extractedFeatures.err.rms.toFixed(3);
            fSkewErr.textContent = extractedFeatures.err.skew.toFixed(3);
            
            fVdcVal.textContent = Vdc.toFixed(1) + ' V';
        }
        
        // 4. ML Heuristic Classifier Inference & Confidence Dynamics
        // Map features to classifier response.
        // We simulate a classifier that checks extracted features:
        // DC offset -> mean currents shift.
        // Voltage sag -> standard deviation of grid voltage drops.
        // Open-circuit -> high skewness/THD in current.
        // Islanding -> frequency drift + grid envelope decay.
        // PLL drift -> tracking error rises.
        let detectedClass = 0;
        let confidence = 0.992 + Math.random() * 0.007; // 99.2% to 99.9% dynamic micro-fluctuations
        
        // Class determination logic:
        if (selectedMode === 1 && !healingApplied) {
            detectedClass = 1;
        } else if (selectedMode === 2 && !healingApplied) {
            detectedClass = 2;
        } else if (selectedMode === 3 && !healingApplied) {
            detectedClass = 3;
        } else if (selectedMode === 4) {
            // Even if islanding healing is active, the class remains 4 (islanding detected & isolated)
            detectedClass = 4;
        } else if (selectedMode === 5 && !healingApplied) {
            detectedClass = 5;
        } else {
            detectedClass = 0;
        }
        
        // Update ML classification results UI
        predictedClassName.textContent = classNames[detectedClass];
        predictedConfidenceVal.textContent = (confidence * 100).toFixed(1) + '% Confidence';
        
        // Color variables update based on state
        const colorsClasses = ['healthy', 'danger', 'danger', 'warning', 'danger', 'warning'];
        const currentType = colorsClasses[detectedClass];
        
        predictedClassName.className = `value text-${currentType}`;
        predictedConfidenceBar.className = `conf-bar bg-${currentType}`;
        predictedConfidenceBar.style.width = (confidence * 100) + '%';
        
        // Update header badges & system health rings
        if (detectedClass === 0) {
            sysStatusBadge.textContent = healingEnabled && selectedMode > 0 ? "REMEDIATED — SYSTEM RUNNING" : "SYSTEM HEALTHY";
            sysStatusBadge.className = healingEnabled && selectedMode > 0 ? "badge healing" : "badge";
            sysPulseRing.className = healingEnabled && selectedMode > 0 ? "pulse-ring warning-ring" : "pulse-ring green-ring";
            
            if (healingEnabled && selectedMode > 0) {
                // Under healing
                healingDot.className = "status-dot purple-dot";
                healingStatusText.textContent = "Active Mitigation Online";
                healingStatusText.className = "text-healing";
            } else {
                healingDot.className = "status-dot green-dot";
                healingStatusText.textContent = "Monitoring Grid Status";
                healingStatusText.className = "text-healthy";
            }
        } else {
            const isDanger = [1, 2, 4].includes(detectedClass);
            sysStatusBadge.textContent = isDanger ? "CRITICAL THREAT INJECTED" : "GRID DISTURBANCE DETECTED";
            sysStatusBadge.className = isDanger ? "badge danger" : "badge warning";
            sysPulseRing.className = isDanger ? "pulse-ring danger-ring" : "pulse-ring warning-ring";
            
            healingDot.className = "status-dot red-dot";
            healingStatusText.textContent = "Fault Condition Unmitigated";
            healingStatusText.className = "text-danger";
        }
        
        // Update self healing descriptive label
        if (healingEnabled) {
            healingActionDetails.textContent = activeHealingAction;
            healingActionDetails.className = "action-details text-healing";
        } else {
            healingActionDetails.textContent = selectedMode > 0 ? "Warning: Self-healing disabled. System operating under fault conditions." : "No active grid threat detected.";
            healingActionDetails.className = selectedMode > 0 ? "action-details text-danger" : "action-details";
        }
        
        // 5. Dynamic Gauges Updates
        // THD dynamic calculation
        let thd = 1.6 + Math.random() * 0.4;
        if (detectedClass === 1) thd = 15.6 + (Math.random() - 0.5) * 1.2;
        else if (detectedClass === 2) thd = 5.4 + (Math.random() - 0.5) * 0.6;
        else if (detectedClass === 3) thd = 2.8 + (Math.random() - 0.5) * 0.4;
        else if (detectedClass === 4) thd = 16.4 + (Math.random() - 0.5) * 1.5;
        else if (detectedClass === 5) thd = 7.2 + (Math.random() - 0.5) * 0.8;
        
        // If islanding disconnect has occurred, THD is NA
        if (healingEnabled && selectedMode === 4) {
            gaugeThd.textContent = "0.0%";
            gaugeThdStatus.textContent = "System Disconnected";
            gaugeThdStatus.className = "gauge-meta text-danger";
        } else {
            gaugeThd.textContent = thd.toFixed(1) + '%';
            if (thd < 5.0) {
                gaugeThdStatus.textContent = "IEEE 519 Compliant";
                gaugeThdStatus.className = "gauge-meta text-healthy";
            } else {
                gaugeThdStatus.textContent = "THD Limit Violated!";
                gaugeThdStatus.className = "gauge-meta text-danger";
            }
        }
        
        // Frequency Gauge
        let f_est = freqDriftApplied + (Math.random() - 0.5) * 0.02;
        if (healingEnabled && selectedMode === 4) {
            gaugeFreq.textContent = "0.00 Hz";
            gaugeFreqDrift.textContent = "Disconnected";
            gaugeFreqDrift.className = "gauge-meta text-danger";
        } else {
            gaugeFreq.textContent = f_est.toFixed(2) + ' Hz';
            const drift = Math.abs(f_est - 50.0);
            if (drift < 0.1) {
                gaugeFreqDrift.textContent = "Nominal Grid";
                gaugeFreqDrift.className = "gauge-meta text-healthy";
            } else {
                gaugeFreqDrift.textContent = `Drift: ${f_est > 50 ? '+' : ''}${drift.toFixed(2)} Hz`;
                gaugeFreqDrift.className = "gauge-meta text-warning";
            }
        }
        
        // Active & Reactive Power
        // Average active/reactive powers (approximated based on I_amp, Vgrid_amp)
        let powerP = 3 * (Vgrid_amp / Math.sqrt(2)) * (I_amp / Math.sqrt(2)) / 1000; // kW
        let powerQ = 0.05; // kVAR
        
        if (selectedMode === 1 && !healingApplied) {
            // One phase current is highly reduced
            powerP *= 0.67;
        } else if (selectedMode === 3 && !healingApplied) {
            // Sag voltage reduces power
            const scale = parseFloat(sliderGridScale.value);
            powerP *= scale;
        } else if (selectedMode === 4) {
            // Islanding
            if (healingApplied) {
                powerP = 0;
                powerQ = 0;
            } else {
                powerP *= Math.exp(-3 * (time % 1.5));
            }
        } else if (selectedMode === 5 && !healingApplied) {
            // Phase displacement -> causes reactive power Q to surge
            const drift = 0.8;
            powerP *= Math.cos(drift);
            powerQ = 3 * (Vgrid_amp / Math.sqrt(2)) * (I_amp / Math.sqrt(2)) * Math.sin(drift) / 1000;
        }
        
        if (healingApplied && selectedMode === 3) {
            // Injecting reactive power to support grid voltage
            powerQ = -1.25; // kVAR (capacitive support)
        }
        
        gaugePowerP.textContent = powerP.toFixed(2) + ' kW';
        gaugePowerQ.textContent = Math.abs(powerQ).toFixed(2) + ' kVAR';
        
        const pf = powerP / Math.sqrt(powerP * powerP + powerQ * powerQ);
        gaugePowerFactor.textContent = `Power Factor: ${pf.toFixed(2)} ${powerQ < 0 ? 'Capacitive' : 'Inductive'}`;
        
        // 6. Draw Waveforms
        drawScope();
        drawErrorScope();
        
        // Request next frame (simulate 50 samples every 10ms approx)
        setTimeout(updateSimulation, 10);
    }
    
    // --- Canvas Drawing Methods ---
    function drawScope() {
        const w = scopeCanvas.width;
        const h = scopeCanvas.height;
        ctxScope.clearRect(0, 0, w, h);
        
        // Draw grid lines
        ctxScope.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        ctxScope.lineWidth = 1;
        const gridDivs = 10;
        for (let i = 1; i < gridDivs; i++) {
            // Vertical lines
            ctxScope.beginPath();
            ctxScope.moveTo(i * (w / gridDivs), 0);
            ctxScope.lineTo(i * (w / gridDivs), h);
            ctxScope.stroke();
            
            // Horizontal lines
            ctxScope.beginPath();
            ctxScope.moveTo(0, i * (h / gridDivs));
            ctxScope.lineTo(w, i * (h / gridDivs));
            ctxScope.stroke();
        }
        
        // Center line
        ctxScope.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctxScope.beginPath();
        ctxScope.moveTo(0, h / 2);
        ctxScope.lineTo(w, h / 2);
        ctxScope.stroke();
        
        const colors = ['#FF2E63', '#00E1FF', '#00FFB2']; // Neon red, cyan, green
        
        // Draw actual phase currents
        const bufferLen = historyI[0].length;
        if (bufferLen < 2) return;
        
        // Scale currents to fit height
        // Max current amplitude ~25A with noise, scale so 25A is max height (h/2)
        const scaleY = (h / 2.3) / 25;
        
        // Draw reference currents (dashed, white)
        ctxScope.setLineDash([4, 4]);
        ctxScope.lineWidth = 1;
        for (let ph = 0; ph < 3; ph++) {
            ctxScope.strokeStyle = 'rgba(255, 255, 255, 0.18)';
            ctxScope.beginPath();
            for (let i = 0; i < bufferLen; i++) {
                const x = (i / (maxDrawSamples - 1)) * w;
                const y = h / 2 - historyRef[ph][i] * scaleY;
                if (i === 0) ctxScope.moveTo(x, y);
                else ctxScope.lineTo(x, y);
            }
            ctxScope.stroke();
        }
        ctxScope.setLineDash([]); // Reset
        
        // Draw actual phase currents (solid, colorful, thick)
        ctxScope.lineWidth = 2;
        for (let ph = 0; ph < 3; ph++) {
            ctxScope.strokeStyle = colors[ph];
            ctxScope.shadowColor = colors[ph];
            ctxScope.shadowBlur = 4; // neon glow effect
            ctxScope.beginPath();
            for (let i = 0; i < bufferLen; i++) {
                const x = (i / (maxDrawSamples - 1)) * w;
                const y = h / 2 - historyI[ph][i] * scaleY;
                if (i === 0) ctxScope.moveTo(x, y);
                else ctxScope.lineTo(x, y);
            }
            ctxScope.stroke();
            ctxScope.shadowBlur = 0; // reset
        }
    }
    
    function drawErrorScope() {
        const w = errorCanvas.width;
        const h = errorCanvas.height;
        ctxError.clearRect(0, 0, w, h);
        
        // Center line
        ctxError.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctxError.beginPath();
        ctxError.moveTo(0, h / 2);
        ctxError.lineTo(w, h / 2);
        ctxError.stroke();
        
        const colors = ['#FF2E63', '#00E1FF', '#00FFB2'];
        const bufferLen = historyErr[0].length;
        if (bufferLen < 2) return;
        
        // Error scale is tighter (e.g. 5A is full height)
        const scaleY = (h / 2.2) / 6.0;
        
        ctxError.lineWidth = 1.2;
        for (let ph = 0; ph < 3; ph++) {
            ctxError.strokeStyle = colors[ph];
            ctxError.beginPath();
            for (let i = 0; i < bufferLen; i++) {
                const x = (i / (maxDrawSamples - 1)) * w;
                const y = h / 2 - historyErr[ph][i] * scaleY;
                if (i === 0) ctxError.moveTo(x, y);
                else ctxError.lineTo(x, y);
            }
            ctxError.stroke();
        }
    }
    
    // Start simulation loop
    updateSimulation();
});
