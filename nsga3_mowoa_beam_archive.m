function RESULTS = nsga3_mowoa_beam_archive()
% NSGA3_MOWOA_BEAM_ARCHIVE
% -------------------------------------------------------------------------
% Self-contained MATLAB code for multi-objective beam-number optimization
% in a 5G/mmWave sectorized network. Two optimizers are compared:
%   1) NSGA-III: Non-dominated Sorting Genetic Algorithm III.
%   2) MOWOA   : Multi-Objective Whale Optimization Algorithm.
%
% The same network model, Monte Carlo optimization data, initial population,
% population size, number of optimization cycles, mutation probability, and
% final evaluation data are used for both algorithms as far as their different
% mechanisms allow. This gives both algorithms the same approximate objective-
% function evaluation budget:
%       population + population * number_of_cycles.
%
% Decision vector:
%   One independently selected beam count for each of the 21 sectors.
%   Candidate beam counts are [2 3 4 6 8 10 15 20 24].
%
% Five optimization objectives:
%   maximize average effective throughput;
%   maximize 5%-tile SINR;
%   maximize Jain fairness index of throughput;
%   minimize beam-management overhead;
%   minimize average inter-sector interference.
%
% Both algorithms maintain two Pareto archives:
%   1) A bounded working archive (Aworking = 70) used during the search.
%      In MOWOA, leaders are selected only from this working archive. NSGA-III
%      retains its native environmental-selection mechanism and keeps the same
%      bounded working archive for symmetric archive tracking.
%   2) An unbounded historical archive containing the unique non-dominated
%      solutions found over all completed evaluations. This historical archive
%      is used for final analysis, equal-weight FoM selection, Pareto figures,
%      Hypervolume-ready storage, and cross-algorithm comparison.
%
% FoM values from separately normalized historical archives are intended for
% within-algorithm selection, not as direct cross-algorithm performance measures.
% Direct comparison is performed using the final metrics evaluated on the same
% 200 Monte Carlo snapshots.
%
% MOWOA implementation:
%   - External bounded working Pareto archive plus unbounded historical archive.
%   - Sparse-region leader selection using the working archive crowding distance.
%   - Standard WOA encircling, global search, and logarithmic spiral updates.
%   - Continuous positions are rounded/clipped to valid candidate indices.
%   - A discrete mutation/perturbation with the same probability as NSGA-III
%     is applied after every whale-position update.
%
% Main comparisons:
%   fixed N=8 beams, fixed N=24 beams (table/reference), NSGA-III best,
%   and MOWOA best. Comparative figures focus on fixed N=8, NSGA-III, MOWOA.
%
% How to run:
%   RESULTS = nsga3_mowoa_beam_archive;
% -------------------------------------------------------------------------

clc; close all;
rng(31001, 'twister');

% -------------------------------------------------------------------------
% User options
% -------------------------------------------------------------------------
saveFigures = true;
saveMatFile = true;
outDir = fullfile(pwd, 'nsga3_mowoa_beam_no_wf_scientific_outputs');
if (saveFigures || saveMatFile) && ~exist(outDir, 'dir')
    mkdir(outDir);
end

% -------------------------------------------------------------------------
% Network and propagation parameters
% -------------------------------------------------------------------------
P.numBS = 7;
P.numSec = 3;
P.numPairs = P.numBS * P.numSec;

P.UEsPerSec = 20;
P.useNonUniformLoad = true;
P.minUEsPerSec = 8;
P.maxUEsPerSec = 35;
P.loadJitterUE = 3;
P.sectorLoadProfile = [34 20 10, ...
                       14 28 12, ...
                       22  9 32, ...
                       11 24 18, ...
                       30 13 21, ...
                       16 27  8, ...
                       25 15 29];

% The same optimization snapshots are used by NSGA-III and MOWOA.
P.NsnapOpt = 25;
P.NsnapEval = 200;

P.BW_Hz = 100e6;
P.f_c_Hz = 28e9;
P.Pt_dBm = 33;
P.Pt_W = 10^((P.Pt_dBm - 30)/10);
P.NF_dB = 9;
P.noise_dBm = -174 + 10*log10(P.BW_Hz) + P.NF_dB;
P.noise_W = 10^((P.noise_dBm - 30)/10);
P.antennaGain_dBi = 15;
P.frontBackLimit_dB = 30;
P.sectorSpan_deg = 120;
P.sectorRadius_m = 150;

% 5G-like SSB/CSI overhead model.
P.mu = 3;
P.nSym_slot = 14;
P.nSym_SSB = 4;
P.T_slot_ms = 1/(2^P.mu);
P.Tsym_ms = P.T_slot_ms/P.nSym_slot;
P.T_SSB_ms = P.nSym_SSB*P.Tsym_ms;
P.T_ssb_period_ms = 20;
P.T_csi_period_ms = 5;
P.OH_CSI_reference = P.Tsym_ms/P.T_csi_period_ms;
P.OH_cap = 0.50;

P.beamComplexityPenaltyMax = 0.10;
P.beamComplexityPenaltyPower = 1.0;

P.rho_active = 0.70;
P.rho_idle = 0.05;
P.Nref = 30;
P.eta_overlap = 0.50;

P.pCorrectAt8 = 0.90;
P.pCorrectSlope = 0.012;
P.pCorrectMin = 0.65;
P.pCorrectMax = 0.96;

P.sigmaSh_dB = 4;
P.coverageThreshold_dB = 3.0;

P.bsPos = [500 500;
           800 500;
           200 500;
           650 760;
           650 240;
           350 760;
           350 240];

P.beamCandidates = [2 3 4 6 8 10 15 20 24];
P.numCandidates = numel(P.beamCandidates);
P.baselineBeams = 8;

% -------------------------------------------------------------------------
% Common multi-objective optimization parameters
% -------------------------------------------------------------------------
P.numObjectives = 5;
P.numDecisionVars = P.numPairs;
P.refDivisions = 4;
P.refPoints = generate_reference_points(P.numObjectives, P.refDivisions);
P.popSize = size(P.refPoints, 1);       % 70 solutions/whales.
P.maxGen = 100;                         % 100 generations/iterations for both.
P.pMutation = min(0.20, 2/P.numDecisionVars);

% NSGA-III-specific parameter.
P.pCrossover = 0.90;

% Common dual-archive parameter. Both algorithms use the same bounded
% working archive size, while their historical archives are unbounded.
P.workingArchiveSize = P.popSize;

% MOWOA-specific parameter. WOA has no genetic crossover; its corresponding
% search mechanisms are encircling, random-whale search, and spiral motion.
P.mowoaSpiralB = 1.0;

% Reproducibility and fair initialization.
P.seedOptData = 41001;
P.seedEvalData = 51001;
P.seedInitialPopulation = 61001;
P.seedNSGAOperators = 62001;
P.seedMOWOAOperators = 63001;

fprintf('\nNSGA-III versus MOWOA beam-number optimization without Water-Filling\n');
fprintf('Network: %d BSs, %d sectors/BS, %d active sectors\n', P.numBS, P.numSec, P.numPairs);
if P.useNonUniformLoad
    fprintf('Non-uniform sector load is enabled. Mean nominal load = %.2f UEs/sector\n', mean(P.sectorLoadProfile));
    fprintf('Approx. UEs/snapshot: %.0f, Opt snapshots: %d, Final evaluation snapshots: %d\n', ...
        sum(P.sectorLoadProfile), P.NsnapOpt, P.NsnapEval);
else
    fprintf('Uniform UEs/snapshot: %d, Opt snapshots: %d, Final evaluation snapshots: %d\n', ...
        P.numPairs*P.UEsPerSec, P.NsnapOpt, P.NsnapEval);
end
fprintf('Beam candidates per sector: '); fprintf('%d ', P.beamCandidates); fprintf('\n');
fprintf('Decision variables: %d sectors, objectives: %d\n', P.numDecisionVars, P.numObjectives);
fprintf('Common population/whales: %d, generations/iterations: %d, mutation probability: %.5f\n', ...
    P.popSize, P.maxGen, P.pMutation);
fprintf('Approximate objective evaluations per algorithm: %d\n', P.popSize*(P.maxGen+1));
fprintf('NSGA-III crossover probability: %.2f, reference points: %d\n', ...
    P.pCrossover, size(P.refPoints,1));
fprintf('Common working archive size: %d; historical archives: unbounded\n', ...
    P.workingArchiveSize);
fprintf('MOWOA spiral constant b: %.2f\n\n', P.mowoaSpiralB);

% -------------------------------------------------------------------------
% Fixed optimization data and identical initial population
% -------------------------------------------------------------------------
fprintf('Generating common optimization Monte Carlo data...\n');
DATAopt = generate_mc_data(P, P.NsnapOpt, P.seedOptData);
initialPop = generate_initial_population(P, P.seedInitialPopulation);

% -------------------------------------------------------------------------
% Run NSGA-III
% -------------------------------------------------------------------------
fprintf('\nRunning NSGA-III...\n');
NSGA = run_nsga3(P, DATAopt, initialPop);

NSGAParetoGenomes = NSGA.historicalArchive.population;
NSGAParetoK = decode_population(NSGAParetoGenomes, P);
NSGAParetoMetrics = NSGA.historicalArchive.metrics;
[NSGAParetoFoM, NSGAParetoNorm] = compute_fom_equal_weights(NSGAParetoMetrics);
[NSGABestFoMOpt, NSGABestIdx] = max(NSGAParetoFoM);
NSGABestGenome = NSGAParetoGenomes(NSGABestIdx, :);
NSGABestK = P.beamCandidates(NSGABestGenome);
NSGAParetoTopTable = make_pareto_top_table(NSGAParetoK, NSGAParetoMetrics, NSGAParetoFoM, 15);
NSGABestBeamTable = make_beam_table(NSGABestK, P);

% -------------------------------------------------------------------------
% Run MOWOA
% -------------------------------------------------------------------------
fprintf('\nRunning MOWOA...\n');
MOWOA = run_mowoa(P, DATAopt, initialPop);

MOWOAParetoGenomes = MOWOA.historicalArchive.population;
MOWOAParetoK = decode_population(MOWOAParetoGenomes, P);
MOWOAParetoMetrics = MOWOA.historicalArchive.metrics;
[MOWOAParetoFoM, MOWOAParetoNorm] = compute_fom_equal_weights(MOWOAParetoMetrics);
[MOWOABestFoMOpt, MOWOABestIdx] = max(MOWOAParetoFoM);
MOWOABestGenome = MOWOAParetoGenomes(MOWOABestIdx, :);
MOWOABestK = P.beamCandidates(MOWOABestGenome);
MOWOAParetoTopTable = make_pareto_top_table(MOWOAParetoK, MOWOAParetoMetrics, MOWOAParetoFoM, 15);
MOWOABestBeamTable = make_beam_table(MOWOABestK, P);

CombinedBestBeamTable = make_combined_beam_table(NSGABestK, MOWOABestK, P);

fprintf('\n================ NSGA-III TOP PARETO SOLUTIONS BY EQUAL-WEIGHT FoM ================\n');
disp(NSGAParetoTopTable);
fprintf('\n================ NSGA-III BEST BEAM COUNT PER SECTOR ================\n');
disp(NSGABestBeamTable);
fprintf('\nNSGA-III best FoM on its historical optimization archive = %.4f\n', NSGABestFoMOpt);
fprintf('NSGA-III best beam vector [BS1-S1, ..., BS7-S3]:\n');
disp(NSGABestK);

fprintf('\n================ MOWOA TOP PARETO SOLUTIONS BY EQUAL-WEIGHT FoM ================\n');
disp(MOWOAParetoTopTable);
fprintf('\n================ MOWOA BEST BEAM COUNT PER SECTOR ================\n');
disp(MOWOABestBeamTable);
fprintf('\nMOWOA best FoM on its historical optimization archive = %.4f\n', MOWOABestFoMOpt);
fprintf('MOWOA best beam vector [BS1-S1, ..., BS7-S3]:\n');
disp(MOWOABestK);

fprintf('\n================ COMBINED SELECTED BEAMS: FIXED 8, NSGA-III, MOWOA ================\n');
disp(CombinedBestBeamTable);

% -------------------------------------------------------------------------
% Final common Monte Carlo evaluation
% -------------------------------------------------------------------------
fprintf('\nGenerating common final-evaluation Monte Carlo data...\n');
DATAeval = generate_mc_data(P, P.NsnapEval, P.seedEvalData);

baselineK = P.baselineBeams * ones(1, P.numPairs);
fixed24K = 24 * ones(1, P.numPairs);
[BaselineMetrics, BaselineRows] = evaluate_solution(baselineK, P, DATAeval, true);
[Fixed24Metrics, Fixed24Rows] = evaluate_solution(fixed24K, P, DATAeval, true);
[NSGABestMetrics, NSGABestRows] = evaluate_solution(NSGABestK, P, DATAeval, true);
[MOWOABestMetrics, MOWOABestRows] = evaluate_solution(MOWOABestK, P, DATAeval, true);

FinalComparisonTable = make_final_comparison_table(BaselineMetrics, Fixed24Metrics, ...
    NSGABestMetrics, MOWOABestMetrics);
ImprovementTable = make_improvement_table(BaselineMetrics, NSGABestMetrics, MOWOABestMetrics);
OptimizationSetupTable = make_optimization_setup_table(P, NSGA, MOWOA);

fprintf('\n================ FINAL EVALUATION: FIXED 8, FIXED 24, NSGA-III, MOWOA ================\n');
disp(FinalComparisonTable);
fprintf('\n================ IMPROVEMENT OVER FIXED 8 BEAMS ================\n');
disp(ImprovementTable);
fprintf('\n================ OPTIMIZATION SETUP AND ARCHIVE SUMMARY ================\n');
disp(OptimizationSetupTable);

% -------------------------------------------------------------------------
% Export only the requested final-comparison table as a Word-readable .doc file
% -------------------------------------------------------------------------
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
write_table_word(FinalComparisonTable, fullfile(outDir, 'Table2_Final_Comparison.doc'), ...
    'Final Evaluation: Fixed 8, Fixed 24, NSGA-III, and MOWOA');
fprintf('\nThe requested final-comparison table was saved as a Word-readable file in: %s\n', outDir);

% -------------------------------------------------------------------------
% Figures: generate only the four requested manuscript figures
% -------------------------------------------------------------------------
if saveFigures
    % Fig. 1: combined historical Pareto-archive projection.
    plot_combined_pareto_scatter(NSGAParetoMetrics, MOWOAParetoMetrics, outDir);

    % Fig. 2: beam count per sector for Fixed 8, Fixed 24, NSGA-III, and MOWOA.
    plot_best_beams_comparison(CombinedBestBeamTable, outDir);

    % Fig. 3: two beam-pattern panels merged into one figure.
    plot_combined_beam_petals(NSGABestBeamTable, MOWOABestBeamTable, ...
        P, DATAeval(1), outDir);

    % Fig. 4: SINR CDF and effective-throughput CDF merged into one figure.
    plot_combined_cdf_figure(BaselineRows, Fixed24Rows, NSGABestRows, MOWOABestRows, outDir);
end

% -------------------------------------------------------------------------
% Store all results
% -------------------------------------------------------------------------
RESULTS = struct();
RESULTS.P = P;
RESULTS.InitialPopulation = initialPop;
RESULTS.NSGA = NSGA;
RESULTS.MOWOA = MOWOA;
RESULTS.NSGAWorkingArchive = NSGA.workingArchive;
RESULTS.NSGAHistoricalArchive = NSGA.historicalArchive;
RESULTS.MOWOAWorkingArchive = MOWOA.workingArchive;
RESULTS.MOWOAHistoricalArchive = MOWOA.historicalArchive;

RESULTS.NSGAParetoK = NSGAParetoK;
RESULTS.NSGAParetoMetrics = NSGAParetoMetrics;
RESULTS.NSGAParetoFoM = NSGAParetoFoM;
RESULTS.NSGAParetoNormalizedObjectives = NSGAParetoNorm;
RESULTS.NSGAParetoTopTable = NSGAParetoTopTable;
RESULTS.NSGABestK = NSGABestK;
RESULTS.NSGABestFoMOpt = NSGABestFoMOpt;
RESULTS.NSGABestBeamTable = NSGABestBeamTable;
RESULTS.NSGABestMetrics = NSGABestMetrics;
RESULTS.NSGABestRows = NSGABestRows;

RESULTS.MOWOAParetoK = MOWOAParetoK;
RESULTS.MOWOAParetoMetrics = MOWOAParetoMetrics;
RESULTS.MOWOAParetoFoM = MOWOAParetoFoM;
RESULTS.MOWOAParetoNormalizedObjectives = MOWOAParetoNorm;
RESULTS.MOWOAParetoTopTable = MOWOAParetoTopTable;
RESULTS.MOWOABestK = MOWOABestK;
RESULTS.MOWOABestFoMOpt = MOWOABestFoMOpt;
RESULTS.MOWOABestBeamTable = MOWOABestBeamTable;
RESULTS.MOWOABestMetrics = MOWOABestMetrics;
RESULTS.MOWOABestRows = MOWOABestRows;

RESULTS.CombinedBestBeamTable = CombinedBestBeamTable;
RESULTS.BaselineMetrics = BaselineMetrics;
RESULTS.Fixed24Metrics = Fixed24Metrics;
RESULTS.BaselineRows = BaselineRows;
RESULTS.Fixed24Rows = Fixed24Rows;
RESULTS.FinalComparisonTable = FinalComparisonTable;
RESULTS.ImprovementTable = ImprovementTable;
RESULTS.OptimizationSetupTable = OptimizationSetupTable;

% Backward-compatible aliases for earlier scripts expecting NSGA-only names.
RESULTS.ParetoK = NSGAParetoK;
RESULTS.ParetoMetrics = NSGAParetoMetrics;
RESULTS.ParetoFoM = NSGAParetoFoM;
RESULTS.ParetoNormalizedObjectives = NSGAParetoNorm;
RESULTS.ParetoTopTable = NSGAParetoTopTable;
RESULTS.BestK = NSGABestK;
RESULTS.BestFoMOpt = NSGABestFoMOpt;
RESULTS.BestBeamTable = NSGABestBeamTable;
RESULTS.BestMetrics = NSGABestMetrics;
RESULTS.BestRows = NSGABestRows;

if saveMatFile
    save(fullfile(outDir, 'NSGA3_MOWOA_BEAM_OPT_RESULTS_no_wf_scientific.mat'), ...
        'RESULTS', '-v7.3');
end

fprintf('\nDone. NSGA-III and MOWOA used the same network model, optimization data, initial population, and final evaluation data.\n');
fprintf('Both algorithms used a working archive of %d and an unbounded historical archive.\n', P.workingArchiveSize);
fprintf('Water-Filling was not used. Only the requested four figures and one Word-readable comparison table are exported; no CSV files are created.\n');
if saveFigures || saveMatFile
    fprintf('Output folder: %s\n', outDir);
end
end

% =========================================================================
% Monte Carlo data generation
% =========================================================================
function DATA = generate_mc_data(P, Nsnap, seedValue)
rng(seedValue, 'twister');
DATA = repmat(struct(), Nsnap, 1);

for snap = 1:Nsnap
    sectorLoad = generate_sector_loads(P);
    N = sum(sectorLoad);
    user = zeros(N, 5); % bs, sector, pair, x, y
    row = 0;

    % UEs are generated inside each active sector. In the corrected version,
    % sector loads are intentionally non-uniform to avoid artificial network
    % symmetry and to make sector-dependent beam selection meaningful.
    for bs = 1:P.numBS
        sx = P.bsPos(bs,1);
        sy = P.bsPos(bs,2);
        for sec = 1:P.numSec
            pair = (bs-1)*P.numSec + sec;
            Nu = sectorLoad(pair);
            az0 = (sec-1) * P.sectorSpan_deg;
            az1 = sec * P.sectorSpan_deg;
            r = sqrt(rand(Nu,1)) * P.sectorRadius_m;
            az = az0 + rand(Nu,1) * (az1 - az0);
            x = sx + r .* cosd(az);
            y = sy + r .* sind(az);
            idx = row + (1:Nu);
            user(idx,:) = [bs*ones(Nu,1), sec*ones(Nu,1), ...
                           pair*ones(Nu,1), x, y];
            row = row + Nu;
        end
    end

    x = user(:,4);
    y = user(:,5);

    localAng = zeros(N, P.numPairs);
    PL_dB = zeros(N, P.numPairs);

    for q = 1:P.numPairs
        [bsq, secq] = pair_to_bssec(q, P);
        xb = P.bsPos(bsq,1);
        yb = P.bsPos(bsq,2);
        th = atan2d(y - yb, x - xb);
        th(th < 0) = th(th < 0) + 360;

        % Corrected interference-angle model:
        % Do NOT clamp out-of-sector directions to 120-eps. Instead, keep
        % the wrapped local angle in [0,360). The antenna pattern then uses
        % wrap180(localAngle - beamCenter), which naturally gives low gain
        % for users outside the sector main angular region.
        thLocal = th - (secq-1)*P.sectorSpan_deg;
        thLocal = mod(thLocal, 360);
        localAng(:,q) = thLocal;

        d = hypot(x - xb, y - yb);
        PL_dB(:,q) = path_loss_uma_simple(d, P.f_c_Hz) + P.sigmaSh_dB*randn(N,1);
    end

    rhoSector = P.rho_idle + (P.rho_active - P.rho_idle) .* min(1, sectorLoad(:)/P.Nref);

    DATA(snap).snapshot = snap;
    DATA(snap).sectorLoad = sectorLoad(:);
    DATA(snap).rhoSector = rhoSector(:);
    DATA(snap).bsIdx = user(:,1);
    DATA(snap).secIdx = user(:,2);
    DATA(snap).pairIdx = user(:,3);
    DATA(snap).x_m = x;
    DATA(snap).y_m = y;
    DATA(snap).localAng = localAng;
    DATA(snap).PL_dB = PL_dB;
    DATA(snap).randCorrect = rand(N,1);
    DATA(snap).randNeighbor = rand(N,1);
    DATA(snap).randInterfBeam = rand(N, P.numPairs);
end
end

function [metrics, rows] = evaluate_solution(Kvec, P, DATA, returnRows)
if nargin < 4
    returnRows = false;
end

Kvec = Kvec(:)';
Nsnap = numel(DATA);
Ntot = 0;
for ss = 1:Nsnap
    Ntot = Ntot + numel(DATA(ss).pairIdx);
end

allSinr_dB = zeros(Ntot,1);
allTput = zeros(Ntot,1);
allInterf_W = zeros(Ntot,1);
allOH = zeros(Ntot,1);

if returnRows
    allSnap = zeros(Ntot,1);
    allBS = zeros(Ntot,1);
    allSec = zeros(Ntot,1);
    allPair = zeros(Ntot,1);
    allK = zeros(Ntot,1);
    allLoad = zeros(Ntot,1);
    allPcorrect = zeros(Ntot,1);
    allX = zeros(Ntot,1);
    allY = zeros(Ntot,1);
end

sectorOH = compute_sector_overheads(Kvec, P);
ptr = 0;

for ss = 1:Nsnap
    D = DATA(ss);
    N = numel(D.pairIdx);
    pairIdx = D.pairIdx(:);
    rowIdx = (1:N)';

    Kserv = Kvec(pairIdx);
    Kserv = Kserv(:);
    HPBWserv = P.sectorSpan_deg ./ Kserv;

    indServ = sub2ind([N P.numPairs], rowIdx, pairIdx);
    thServ = D.localAng(indServ);

    % Serving users are generated inside their serving sector, but we still
    % clamp numerically to [0,120) for robust beam-index calculation.
    thServForIndex = thServ;
    thServForIndex(thServForIndex < 0) = 0;
    thServForIndex(thServForIndex >= P.sectorSpan_deg) = P.sectorSpan_deg - eps;

    PLserv = D.PL_dB(indServ);

    % Uniform sector codebook: center angles are (0.5:1:K-0.5)*HPBW.
    bestBeamIdx = floor(thServForIndex ./ HPBWserv) + 1;
    bestBeamIdx = max(bestBeamIdx, 1);
    bestBeamIdx = min(bestBeamIdx, Kserv);

    chosenBeamIdx = bestBeamIdx;

    % Corrected beam-selection model: p_correct is K-dependent. Narrower beams
    % are more sensitive to angular error and therefore have a lower chance of
    % selecting the exact best beam.
    pCorrectUser = beam_correct_probability(Kserv, P);
    miss = D.randCorrect(:) > pCorrectUser;
    if any(miss)
        onlyLeft = miss & (bestBeamIdx == Kserv) & (Kserv > 1);
        onlyRight = miss & (bestBeamIdx == 1) & (Kserv > 1);
        middle = miss & (bestBeamIdx > 1) & (bestBeamIdx < Kserv);

        chosenBeamIdx(onlyLeft) = bestBeamIdx(onlyLeft) - 1;
        chosenBeamIdx(onlyRight) = bestBeamIdx(onlyRight) + 1;

        midIdx = find(middle);
        if ~isempty(midIdx)
            chosenBeamIdx(midIdx) = bestBeamIdx(midIdx) - 1;
            chooseRight = D.randNeighbor(midIdx) >= 0.5;
            chosenBeamIdx(midIdx(chooseRight)) = bestBeamIdx(midIdx(chooseRight)) + 1;
        end
    end

    beamCenterServ = (chosenBeamIdx - 0.5) .* HPBWserv;
    deltaServ = abs(wrap180_local(thServ - beamCenterServ));
    Gserv = beam_gain(deltaServ, HPBWserv, P.antennaGain_dBi, P.frontBackLimit_dB);
    g_eff = 10.^((Gserv - PLserv)/10);

    % Load-aware inter-sector interference from all non-serving sectors.
    % Each interfering sector has its own rho based on its current snapshot load.
    I_W = zeros(N,1);
    for q = 1:P.numPairs
        active = pairIdx ~= q;
        if ~any(active)
            continue;
        end
        Kq = Kvec(q);
        HPBWq = P.sectorSpan_deg / Kq;
        intBeamIdx = floor(D.randInterfBeam(:,q) * Kq) + 1;
        intBeamIdx = max(intBeamIdx, 1);
        intBeamIdx = min(intBeamIdx, Kq);
        beamCenterI = (intBeamIdx - 0.5) * HPBWq;

        % Corrected: localAng can be outside [0,120). The wrapped angular
        % distance to the selected interfering beam center is used directly.
        deltaI = abs(wrap180_local(D.localAng(:,q) - beamCenterI));
        GI = beam_gain(deltaI, HPBWq, P.antennaGain_dBi, P.frontBackLimit_dB);
        Iterm = P.eta_overlap * D.rhoSector(q) * 10.^((P.Pt_dBm + GI - D.PL_dB(:,q) - 30)/10);
        I_W(active) = I_W(active) + Iterm(active);
    end

    sinrLinear = P.Pt_W * g_eff ./ (P.noise_W + I_W);
    sinr_dB = 10*log10(max(sinrLinear, realmin));
    OHuser = sectorOH(pairIdx);

    % No WF: equal resource share among the users currently assigned to the
    % same sector. With non-uniform load, bandwidth share is sector-dependent.
    loadUser = D.sectorLoad(pairIdx);
    loadUser = max(loadUser(:), 1);
    tput = (P.BW_Hz ./ loadUser) .* log2(1 + sinrLinear) / 1e6 .* (1 - OHuser);

    idxOut = ptr + (1:N);
    allSinr_dB(idxOut) = sinr_dB;
    allTput(idxOut) = tput;
    allInterf_W(idxOut) = I_W;
    allOH(idxOut) = OHuser;

    if returnRows
        allSnap(idxOut) = ss;
        allBS(idxOut) = D.bsIdx;
        allSec(idxOut) = D.secIdx;
        allPair(idxOut) = pairIdx;
        allK(idxOut) = Kserv;
        allLoad(idxOut) = loadUser;
        allPcorrect(idxOut) = pCorrectUser;
        allX(idxOut) = D.x_m;
        allY(idxOut) = D.y_m;
    end

    ptr = ptr + N;
end

metrics = struct();
metrics.AvgSINR_dB = mean(allSinr_dB);
metrics.P5SINR_dB = prctile(allSinr_dB, 5);
metrics.Coverage_pct = 100 * mean(allSinr_dB > P.coverageThreshold_dB);
metrics.AvgEffTput_Mbps = mean(allTput);
metrics.JFI = jain_fairness(allTput);
metrics.AvgOverhead = mean(allOH);
metrics.AvgOverhead_percent = 100 * metrics.AvgOverhead;
metrics.AvgInterference_W = mean(allInterf_W);
metrics.AvgInterference_dBm = 10*log10(max(metrics.AvgInterference_W, realmin)*1000);
metrics.P5Tput_Mbps = prctile(allTput, 5);
metrics.MedianTput_Mbps = median(allTput);

if returnRows
    interf_dBm = 10*log10(max(allInterf_W, realmin)*1000);
    rows = table(allSnap, allBS, allSec, allPair, allK, allLoad, allPcorrect, allX, allY, ...
        allSinr_dB, allTput, allOH, 100*allOH, allInterf_W, interf_dBm, ...
        'VariableNames', {'snapshot','bs','sector','pair','numBeams','sectorLoad','pCorrectBeam','x_m','y_m', ...
        'sinr_dB','tput_Mbps','overhead','overhead_percent','interference_W','interference_dBm'});
else
    rows = [];
end
end

function sectorOH = compute_sector_overheads(Kvec, P)
Kvec = Kvec(:)';

% Corrected overhead model:
% 1) SSB sweeping overhead grows with K.
% 2) CSI/beam refinement overhead also grows with K, instead of being fixed.
% 3) A normalized beam-management complexity penalty is included to represent
%    search latency, measurement/reporting burden, and control complexity.
OH_SSB = Kvec * P.T_SSB_ms / P.T_ssb_period_ms;
OH_CSI = Kvec * P.Tsym_ms / P.T_csi_period_ms;

Kmin = min(P.beamCandidates);
Kmax = max(P.beamCandidates);
if Kmax > Kmin
    Knorm = (Kvec - Kmin) / (Kmax - Kmin);
else
    Knorm = zeros(size(Kvec));
end
OH_complexity = P.beamComplexityPenaltyMax * (Knorm .^ P.beamComplexityPenaltyPower);

sectorOH = min(OH_SSB + OH_CSI + OH_complexity, P.OH_cap);
sectorOH = sectorOH(:);
end

function pop = generate_initial_population(P, seedValue)
% Identical initial genomes are supplied to NSGA-III and MOWOA.
rng(seedValue, 'twister');
pop = randi(P.numCandidates, P.popSize, P.numDecisionVars);
for c = 1:min(P.numCandidates, P.popSize)
    pop(c,:) = c;
end
end

function NSGA = run_nsga3(P, DATAopt, initialPop)
D = P.numDecisionVars;
N = P.popSize;

rng(P.seedNSGAOperators, 'twister');
pop = initialPop;
if size(pop,1) ~= N || size(pop,2) ~= D
    error('NSGA-III initial population dimensions do not match P.popSize and P.numDecisionVars.');
end

[obj, metrics] = evaluate_population(pop, P, DATAopt);
[~, rank] = non_dominated_sort(obj);

% Dual archives:
%   historicalArchive: unbounded global non-dominated set over all evaluations.
%   workingArchive   : bounded crowding-distance subset, Aworking = 70.
[historicalPop, historicalObj, historicalMetrics] = update_pareto_archive( ...
    [], [], [], pop, obj, metrics);
[workingPop, workingObj, workingMetrics] = truncate_pareto_archive( ...
    historicalPop, historicalObj, historicalMetrics, P.workingArchiveSize);

history = initialize_optimizer_history(P.maxGen);

for gen = 1:P.maxGen
    matingPool = tournament_selection(pop, rank, obj, N);
    offspring = make_offspring(matingPool, P);
    [objOff, metricsOff] = evaluate_population(offspring, P, DATAopt);

    [historicalPop, historicalObj, historicalMetrics] = update_pareto_archive( ...
        historicalPop, historicalObj, historicalMetrics, ...
        offspring, objOff, metricsOff);
    [workingPop, workingObj, workingMetrics] = truncate_pareto_archive( ...
        historicalPop, historicalObj, historicalMetrics, P.workingArchiveSize);

    combPop = [pop; offspring];
    combObj = [obj; objOff];
    combMetrics = [metrics; metricsOff];

    [pop, obj, metrics, rank] = environmental_selection_nsga3( ...
        combPop, combObj, combMetrics, N, P.refPoints);

    history = update_optimizer_history(history, gen, rank, metrics, ...
        workingPop, workingMetrics, historicalPop, historicalMetrics);

    if mod(gen,5) == 0 || gen == 1 || gen == P.maxGen
        fprintf(['NSGA-III generation %3d/%3d | Pareto=%3d | Working=%3d | ' ...
            'Historical=%3d | Best Historical FoM=%.4f | Best AvgTput=%.3f Mbps | ' ...
            'Best P5SINR=%.3f dB\n'], ...
            gen, P.maxGen, history.ParetoCount(gen), ...
            history.WorkingArchiveCount(gen), history.HistoricalArchiveCount(gen), ...
            history.BestHistoricalArchiveFoM(gen), history.BestAvgTput_Mbps(gen), ...
            history.BestP5SINR_dB(gen));
    end
end

NSGA = struct();
NSGA.population = pop;
NSGA.Kpopulation = decode_population(pop, P);
NSGA.objectives = obj;
NSGA.metrics = metrics;
NSGA.rank = rank;

NSGA.workingArchive = struct();
NSGA.workingArchive.population = workingPop;
NSGA.workingArchive.Kpopulation = decode_population(workingPop, P);
NSGA.workingArchive.objectives = workingObj;
NSGA.workingArchive.metrics = workingMetrics;
NSGA.workingArchive.limit = P.workingArchiveSize;

NSGA.historicalArchive = struct();
NSGA.historicalArchive.population = historicalPop;
NSGA.historicalArchive.Kpopulation = decode_population(historicalPop, P);
NSGA.historicalArchive.objectives = historicalObj;
NSGA.historicalArchive.metrics = historicalMetrics;
NSGA.historicalArchive.limit = inf;

% Backward-compatible alias: final analysis archive is the historical archive.
NSGA.archive = NSGA.historicalArchive;
NSGA.history = history;
NSGA.metricColumns = {'AvgEffTput_Mbps','P5SINR_dB','JFI','AvgOverhead_percent','AvgInterference_dBm','Coverage_pct','AvgSINR_dB','P5Tput_Mbps'};
NSGA.objectiveColumns = {'min_negative_AvgEffTput','min_negative_P5SINR','min_negative_JFI','min_AvgOverhead_percent','min_AvgInterference_dBm'};
NSGA.approximateEvaluations = P.popSize*(P.maxGen+1);
end

function MOWOA = run_mowoa(P, DATAopt, initialPop)
% Multi-Objective Whale Optimization Algorithm for discrete candidate indices.
N = P.popSize;
D = P.numDecisionVars;

rng(P.seedMOWOAOperators, 'twister');
positions = double(initialPop);
pop = discretize_whale_positions(positions, P);
[obj, metrics] = evaluate_population(pop, P, DATAopt);

% Dual archives:
%   historicalArchive: unbounded global non-dominated set over all evaluations.
%   workingArchive   : bounded archive used exclusively for leader selection.
[historicalPop, historicalObj, historicalMetrics] = update_pareto_archive( ...
    [], [], [], pop, obj, metrics);
[workingPop, workingObj, workingMetrics] = truncate_pareto_archive( ...
    historicalPop, historicalObj, historicalMetrics, P.workingArchiveSize);

history = initialize_optimizer_history(P.maxGen);

for iter = 1:P.maxGen
    if P.maxGen > 1
        progress = (iter-1)/(P.maxGen-1);
    else
        progress = 1;
    end
    a = 2 - 2*progress;          % Exploration-to-exploitation coefficient.
    a2 = -1 - progress;          % Standard spiral l-range decreases -1 to -2.

    newPositions = zeros(N,D);
    for i = 1:N
        % Leaders are selected only from the bounded working archive.
        leaderGenome = select_mowoa_leader(workingPop, workingObj);
        leaderPosition = double(leaderGenome);
        currentPosition = positions(i,:);

        r1 = rand;
        r2 = rand;
        A = 2*a*r1 - a;
        C = 2*r2;
        p = rand;

        if p < 0.5
            if abs(A) < 1
                % Encircling/exploitation toward a non-dominated archive leader.
                distance = abs(C*leaderPosition - currentPosition);
                candidatePosition = leaderPosition - A*distance;
            else
                % Global exploration around another randomly selected whale.
                randomIndex = randi(N);
                randomPosition = positions(randomIndex,:);
                distance = abs(C*randomPosition - currentPosition);
                candidatePosition = randomPosition - A*distance;
            end
        else
            % Logarithmic spiral exploitation around the archive leader.
            distance = abs(leaderPosition - currentPosition);
            l = (a2 - 1)*rand + 1;
            candidatePosition = distance .* exp(P.mowoaSpiralB*l) .* ...
                cos(2*pi*l) + leaderPosition;
        end

        newPositions(i,:) = candidatePosition;
    end

    % Bound, discretize, and apply the same per-variable mutation probability
    % used in NSGA-III. Mutation here is a discrete diversification operator,
    % not a replacement for the native WOA movement equations.
    newPositions = max(1, min(P.numCandidates, newPositions));
    pop = discretize_whale_positions(newPositions, P);
    for i = 1:N
        pop(i,:) = mutate_integer_genome(pop(i,:), P);
    end
    positions = double(pop);

    [obj, metrics] = evaluate_population(pop, P, DATAopt);
    [historicalPop, historicalObj, historicalMetrics] = update_pareto_archive( ...
        historicalPop, historicalObj, historicalMetrics, pop, obj, metrics);
    [workingPop, workingObj, workingMetrics] = truncate_pareto_archive( ...
        historicalPop, historicalObj, historicalMetrics, P.workingArchiveSize);
    [~, rank] = non_dominated_sort(obj);

    history = update_optimizer_history(history, iter, rank, metrics, ...
        workingPop, workingMetrics, historicalPop, historicalMetrics);

    if mod(iter,5) == 0 || iter == 1 || iter == P.maxGen
        fprintf(['MOWOA iteration  %3d/%3d | Pareto=%3d | Working=%3d | ' ...
            'Historical=%3d | Best Historical FoM=%.4f | Best AvgTput=%.3f Mbps | ' ...
            'Best P5SINR=%.3f dB\n'], ...
            iter, P.maxGen, history.ParetoCount(iter), ...
            history.WorkingArchiveCount(iter), history.HistoricalArchiveCount(iter), ...
            history.BestHistoricalArchiveFoM(iter), history.BestAvgTput_Mbps(iter), ...
            history.BestP5SINR_dB(iter));
    end
end

MOWOA = struct();
MOWOA.population = pop;
MOWOA.Kpopulation = decode_population(pop, P);
MOWOA.objectives = obj;
MOWOA.metrics = metrics;
MOWOA.rank = rank;

MOWOA.workingArchive = struct();
MOWOA.workingArchive.population = workingPop;
MOWOA.workingArchive.Kpopulation = decode_population(workingPop, P);
MOWOA.workingArchive.objectives = workingObj;
MOWOA.workingArchive.metrics = workingMetrics;
MOWOA.workingArchive.limit = P.workingArchiveSize;

MOWOA.historicalArchive = struct();
MOWOA.historicalArchive.population = historicalPop;
MOWOA.historicalArchive.Kpopulation = decode_population(historicalPop, P);
MOWOA.historicalArchive.objectives = historicalObj;
MOWOA.historicalArchive.metrics = historicalMetrics;
MOWOA.historicalArchive.limit = inf;

% Backward-compatible alias: final analysis archive is the historical archive.
MOWOA.archive = MOWOA.historicalArchive;
MOWOA.history = history;
MOWOA.metricColumns = {'AvgEffTput_Mbps','P5SINR_dB','JFI','AvgOverhead_percent','AvgInterference_dBm','Coverage_pct','AvgSINR_dB','P5Tput_Mbps'};
MOWOA.objectiveColumns = {'min_negative_AvgEffTput','min_negative_P5SINR','min_negative_JFI','min_AvgOverhead_percent','min_AvgInterference_dBm'};
MOWOA.approximateEvaluations = P.popSize*(P.maxGen+1);
end

function pop = discretize_whale_positions(positions, P)
pop = round(positions);
pop = max(1, min(P.numCandidates, pop));
pop = double(pop);
end

function leader = select_mowoa_leader(archivePop, archiveObj)
% Select leaders preferentially from sparse regions of the Pareto archive.
if isempty(archivePop)
    error('MOWOA archive is empty; a leader cannot be selected.');
end
if size(archivePop,1) == 1
    leader = archivePop(1,:);
    return;
end

crowd = crowding_distance_multiobjective(archiveObj);
finiteCrowd = crowd(isfinite(crowd));
if isempty(finiteCrowd)
    weights = ones(size(crowd));
else
    maxFinite = max(finiteCrowd);
    if maxFinite <= 0
        maxFinite = 1;
    end
    crowd(~isfinite(crowd)) = 2*maxFinite;
    crowd(crowd < 0) = 0;
    weights = crowd + 1e-12;
end

weights = weights / sum(weights);
r = rand;
cumulative = cumsum(weights);
idx = find(cumulative >= r, 1, 'first');
if isempty(idx)
    idx = numel(weights);
end
leader = archivePop(idx,:);
end

function [workingPop, workingObj, workingMetrics] = truncate_pareto_archive( ...
    historicalPop, historicalObj, historicalMetrics, maxSize)
% TRUNCATE_PARETO_ARCHIVE
% Creates the bounded working archive from the complete historical Pareto
% archive. Crowding distance preserves sparse and boundary regions.
workingPop = historicalPop;
workingObj = historicalObj;
workingMetrics = historicalMetrics;

if size(workingPop,1) > maxSize
    crowd = crowding_distance_multiobjective(workingObj);
    [~, order] = sort(crowd, 'descend');
    keep = order(1:maxSize);
    workingPop = workingPop(keep,:);
    workingObj = workingObj(keep,:);
    workingMetrics = workingMetrics(keep,:);
end
end

function distance = crowding_distance_multiobjective(obj)
N = size(obj,1);
M = size(obj,2);
distance = zeros(N,1);
if N <= 2
    distance(:) = inf;
    return;
end

for m = 1:M
    [values, order] = sort(obj(:,m));
    distance(order(1)) = inf;
    distance(order(end)) = inf;
    rangeValue = values(end) - values(1);
    if abs(rangeValue) < eps
        continue;
    end
    for i = 2:N-1
        if isfinite(distance(order(i)))
            distance(order(i)) = distance(order(i)) + ...
                (values(i+1) - values(i-1))/rangeValue;
        end
    end
end
end

function H = initialize_optimizer_history(nCycles)
H = struct();
H.BestFoM = zeros(nCycles,1);
H.ParetoCount = zeros(nCycles,1);
H.BestAvgTput_Mbps = zeros(nCycles,1);
H.BestP5SINR_dB = zeros(nCycles,1);
H.BestJFI = zeros(nCycles,1);
H.MinOverhead_percent = zeros(nCycles,1);
H.MinInterference_dBm = zeros(nCycles,1);
H.WorkingArchiveCount = zeros(nCycles,1);
H.HistoricalArchiveCount = zeros(nCycles,1);
H.BestWorkingArchiveFoM = zeros(nCycles,1);
H.BestHistoricalArchiveFoM = zeros(nCycles,1);

% Backward-compatible aliases populated during each update.
H.ArchiveCount = zeros(nCycles,1);
H.BestArchiveFoM = zeros(nCycles,1);
end

function H = update_optimizer_history(H, cycle, rank, metrics, ...
    workingPop, workingMetrics, historicalPop, historicalMetrics)
pf = rank == 1;
H.ParetoCount(cycle) = sum(pf);
H.BestAvgTput_Mbps(cycle) = max(metrics(:,1));
H.BestP5SINR_dB(cycle) = max(metrics(:,2));
H.BestJFI(cycle) = max(metrics(:,3));
H.MinOverhead_percent(cycle) = min(metrics(:,4));
H.MinInterference_dBm(cycle) = min(metrics(:,5));

if any(pf)
    f = compute_fom_equal_weights(metrics(pf,:));
    H.BestFoM(cycle) = max(f);
end

H.WorkingArchiveCount(cycle) = size(workingPop,1);
H.HistoricalArchiveCount(cycle) = size(historicalPop,1);

if ~isempty(workingMetrics)
    fWorking = compute_fom_equal_weights(workingMetrics);
    H.BestWorkingArchiveFoM(cycle) = max(fWorking);
end
if ~isempty(historicalMetrics)
    fHistorical = compute_fom_equal_weights(historicalMetrics);
    H.BestHistoricalArchiveFoM(cycle) = max(fHistorical);
end

% Backward-compatible aliases refer to the bounded working archive.
H.ArchiveCount(cycle) = H.WorkingArchiveCount(cycle);
H.BestArchiveFoM(cycle) = H.BestWorkingArchiveFoM(cycle);
end

function [archivePop, archiveObj, archiveMetrics] = update_pareto_archive(archivePop, archiveObj, archiveMetrics, newPop, newObj, newMetrics)
% UPDATE_PARETO_ARCHIVE
% Keeps only unique non-dominated genomes among the previous archive and the
% newly evaluated solutions. Objectives are minimization objectives.
if isempty(archivePop)
    allPop = newPop;
    allObj = newObj;
    allMetrics = newMetrics;
else
    allPop = [archivePop; newPop];
    allObj = [archiveObj; newObj];
    allMetrics = [archiveMetrics; newMetrics];
end

% Remove duplicate genomes to keep the archive compact and deterministic.
[~, ia] = unique(allPop, 'rows', 'stable');
allPop = allPop(ia,:);
allObj = allObj(ia,:);
allMetrics = allMetrics(ia,:);

[~, arank] = non_dominated_sort(allObj);
keep = arank == 1;
archivePop = allPop(keep,:);
archiveObj = allObj(keep,:);
archiveMetrics = allMetrics(keep,:);
end

function [obj, metrics] = evaluate_population(pop, P, DATAopt)
N = size(pop,1);
obj = zeros(N, P.numObjectives);
metrics = zeros(N, 8);

for i = 1:N
    Kvec = P.beamCandidates(pop(i,:));
    m = evaluate_solution(Kvec, P, DATAopt, false);
    metrics(i,:) = [m.AvgEffTput_Mbps, m.P5SINR_dB, m.JFI, ...
                    m.AvgOverhead_percent, m.AvgInterference_dBm, ...
                    m.Coverage_pct, m.AvgSINR_dB, m.P5Tput_Mbps];

    % NSGA-III implementation below is formulated as minimization.
    % Maximization objectives are multiplied by -1.
    obj(i,:) = [-m.AvgEffTput_Mbps, -m.P5SINR_dB, -m.JFI, ...
                 m.AvgOverhead_percent, m.AvgInterference_dBm];
end
end

function Kpop = decode_population(pop, P)
Kpop = P.beamCandidates(pop);
end

% =========================================================================
% NSGA-III operators
% =========================================================================
function matingPool = tournament_selection(pop, rank, obj, N)
D = size(pop,2);
matingPool = zeros(N,D);

% Normalized-objective tie-breaker:
% The five objectives have different numerical scales, so raw summation can
% bias parent selection toward the objective with the largest magnitude.
% Normalizing here reduces premature convergence caused by scale dominance.
normObj = normalize_objectives(obj);

for i = 1:N
    a = randi(N);
    b = randi(N);
    if rank(a) < rank(b)
        winner = a;
    elseif rank(b) < rank(a)
        winner = b;
    else
        % Tie-breaker: smaller aggregate normalized objective value.
        za = sum(normObj(a,:));
        zb = sum(normObj(b,:));
        if za < zb
            winner = a;
        elseif zb < za
            winner = b;
        else
            if rand < 0.5
                winner = a;
            else
                winner = b;
            end
        end
    end
    matingPool(i,:) = pop(winner,:);
end
end

function offspring = make_offspring(parents, P)
N = size(parents,1);
D = size(parents,2);
offspring = zeros(N,D);

order = randperm(N);
parents = parents(order,:);

for i = 1:2:N
    p1 = parents(i,:);
    if i+1 <= N
        p2 = parents(i+1,:);
    else
        p2 = parents(randi(N),:);
    end

    c1 = p1;
    c2 = p2;
    if rand < P.pCrossover
        mask = rand(1,D) < 0.5;
        c1(mask) = p2(mask);
        c2(mask) = p1(mask);
    end

    c1 = mutate_integer_genome(c1, P);
    c2 = mutate_integer_genome(c2, P);

    offspring(i,:) = c1;
    if i+1 <= N
        offspring(i+1,:) = c2;
    end
end
end

function g = mutate_integer_genome(g, P)
for j = 1:numel(g)
    if rand < P.pMutation
        oldVal = g(j);
        newVal = randi(P.numCandidates);
        if P.numCandidates > 1
            while newVal == oldVal
                newVal = randi(P.numCandidates);
            end
        end
        g(j) = newVal;
    end
end
end

function [newPop, newObj, newMetrics, newRank] = environmental_selection_nsga3(pop, obj, metrics, N, refPoints)
[fronts, rank] = non_dominated_sort(obj);
selected = [];
lastFront = [];

for f = 1:numel(fronts)
    fr = fronts{f};
    if numel(selected) + numel(fr) <= N
        selected = [selected; fr(:)]; %#ok<AGROW>
    else
        lastFront = fr(:);
        break;
    end
end

if numel(selected) == N
    finalIdx = selected;
else
    remaining = N - numel(selected);
    normObj = normalize_objectives(obj);
    [assocRef, distRef] = associate_to_reference_points(normObj, refPoints);

    nicheCount = zeros(size(refPoints,1),1);
    for i = 1:numel(selected)
        nicheCount(assocRef(selected(i))) = nicheCount(assocRef(selected(i))) + 1;
    end

    candidates = lastFront(:);
    chosenFromLast = [];

    while numel(chosenFromLast) < remaining && ~isempty(candidates)
        candidateRefs = unique(assocRef(candidates));
        counts = nicheCount(candidateRefs);
        minCount = min(counts);
        refPool = candidateRefs(counts == minCount);
        refChosen = refPool(randi(numel(refPool)));

        candForRef = candidates(assocRef(candidates) == refChosen);
        if isempty(candForRef)
            candidates = setdiff(candidates, candForRef);
            continue;
        end

        if nicheCount(refChosen) == 0
            [~, jj] = min(distRef(candForRef));
            pick = candForRef(jj);
        else
            pick = candForRef(randi(numel(candForRef)));
        end

        chosenFromLast = [chosenFromLast; pick]; %#ok<AGROW>
        nicheCount(refChosen) = nicheCount(refChosen) + 1;
        candidates(candidates == pick) = [];
    end

    if numel(chosenFromLast) < remaining && ~isempty(candidates)
        extraNeed = remaining - numel(chosenFromLast);
        rp = randperm(numel(candidates), min(extraNeed, numel(candidates)));
        chosenFromLast = [chosenFromLast; candidates(rp)]; %#ok<AGROW>
    end

    finalIdx = [selected; chosenFromLast(:)];
end

newPop = pop(finalIdx,:);
newObj = obj(finalIdx,:);
newMetrics = metrics(finalIdx,:);
[~, newRank] = non_dominated_sort(newObj);
end

function normObj = normalize_objectives(obj)
zmin = min(obj, [], 1);
shifted = bsxfun(@minus, obj, zmin);
zmax = max(shifted, [], 1);
zmax(zmax < eps) = 1;
normObj = bsxfun(@rdivide, shifted, zmax);
normObj(~isfinite(normObj)) = 0;
end

function [assocRef, distRef] = associate_to_reference_points(normObj, refPoints)
N = size(normObj,1);
R = size(refPoints,1);
assocRef = zeros(N,1);
distRef = zeros(N,1);

refNorm = sqrt(sum(refPoints.^2,2));
refNorm(refNorm < eps) = 1;
refUnit = bsxfun(@rdivide, refPoints, refNorm);

for i = 1:N
    y = normObj(i,:);
    bestD = inf;
    bestR = 1;
    for r = 1:R
        u = refUnit(r,:);
        proj = sum(y .* u) * u;
        d = sqrt(sum((y - proj).^2));
        if d < bestD
            bestD = d;
            bestR = r;
        end
    end
    assocRef(i) = bestR;
    distRef(i) = bestD;
end
end

function [fronts, rank] = non_dominated_sort(obj)
N = size(obj,1);
S = cell(N,1);
n = zeros(N,1);
rank = inf(N,1);
fronts = {};
F1 = [];

for p = 1:N
    S{p} = [];
    n(p) = 0;
    for q = 1:N
        if p == q
            continue;
        end
        if dominates(obj(p,:), obj(q,:))
            S{p} = [S{p} q]; %#ok<AGROW>
        elseif dominates(obj(q,:), obj(p,:))
            n(p) = n(p) + 1;
        end
    end
    if n(p) == 0
        rank(p) = 1;
        F1 = [F1 p]; %#ok<AGROW>
    end
end

fronts{1} = F1;
i = 1;
while ~isempty(fronts{i})
    Q = [];
    for pp = 1:numel(fronts{i})
        p = fronts{i}(pp);
        for qq = 1:numel(S{p})
            q = S{p}(qq);
            n(q) = n(q) - 1;
            if n(q) == 0
                rank(q) = i + 1;
                Q = [Q q]; %#ok<AGROW>
            end
        end
    end
    i = i + 1;
    fronts{i} = Q; %#ok<AGROW>
end
if isempty(fronts{end})
    fronts(end) = [];
end
end

function tf = dominates(a, b)
tf = all(a <= b) && any(a < b);
end

function refPoints = generate_reference_points(M, H)
% Das and Dennis simplex-lattice reference points.
combs = generate_compositions(H, M);
refPoints = combs / H;
refPoints(sum(refPoints,2) == 0, :) = [];
end

function C = generate_compositions(n, k)
if k == 1
    C = n;
else
    C = [];
    for i = 0:n
        sub = generate_compositions(n-i, k-1);
        C = [C; [i*ones(size(sub,1),1), sub]]; %#ok<AGROW>
    end
end
end

% =========================================================================
% Figure of Merit and tables
% =========================================================================
function [FoM, normMetrics] = compute_fom_equal_weights(metrics)
% metrics columns:
% 1 AvgEffTput_Mbps        benefit
% 2 P5SINR_dB              benefit
% 3 JFI                    benefit
% 4 AvgOverhead_percent    cost
% 5 AvgInterference_dBm    cost
% 6 Coverage_pct           reporting only
% 7 AvgSINR_dB             reporting only
% 8 P5Tput_Mbps            reporting only

X = metrics(:,1:5);
benefit = [true true true false false];
normMetrics = zeros(size(X));

for j = 1:5
    x = X(:,j);
    xmin = min(x);
    xmax = max(x);
    if abs(xmax - xmin) < 1e-12
        normMetrics(:,j) = ones(size(x));
    else
        if benefit(j)
            normMetrics(:,j) = (x - xmin) / (xmax - xmin);
        else
            normMetrics(:,j) = (xmax - x) / (xmax - xmin);
        end
    end
end

weights = ones(1,5)/5;
FoM = normMetrics * weights(:);
end

function T = make_pareto_top_table(Kmat, metrics, FoM, maxRows)
[FoMsort, ord] = sort(FoM, 'descend');
ord = ord(1:min(maxRows, numel(ord)));
FoMsort = FoMsort(1:numel(ord));
Ktop = Kmat(ord,:);
M = metrics(ord,:);
Solution = (1:numel(ord))';
K_vector = cell(numel(ord),1);
MeanBeams = zeros(numel(ord),1);
MinBeams = zeros(numel(ord),1);
MaxBeams = zeros(numel(ord),1);
for i = 1:numel(ord)
    K_vector{i} = mat2str(Ktop(i,:));
    MeanBeams(i) = mean(Ktop(i,:));
    MinBeams(i) = min(Ktop(i,:));
    MaxBeams(i) = max(Ktop(i,:));
end

T = table(Solution, FoMsort, M(:,1), M(:,2), M(:,3), M(:,4), M(:,5), M(:,6), M(:,7), ...
    MeanBeams, MinBeams, MaxBeams, K_vector, ...
    'VariableNames', {'RankByFoM','FoM','AvgEffTput_Mbps','P5SINR_dB','JFI', ...
    'AvgOverhead_percent','AvgInterference_dBm','Coverage_pct','AvgSINR_dB', ...
    'MeanBeams','MinBeams','MaxBeams','K_vector'});
end

function T = make_beam_table(Kvec, P)
Pair = (1:P.numPairs)';
BS = zeros(P.numPairs,1);
Sector = zeros(P.numPairs,1);
for q = 1:P.numPairs
    [BS(q), Sector(q)] = pair_to_bssec(q, P);
end
NumBeams = Kvec(:);
Beamwidth_deg = P.sectorSpan_deg ./ NumBeams;
PcorrectBeam = beam_correct_probability(NumBeams, P);
OH = compute_sector_overheads(Kvec, P);
Overhead_percent = 100 * OH;
T = table(Pair, BS, Sector, NumBeams, Beamwidth_deg, PcorrectBeam, Overhead_percent, ...
    'VariableNames', {'Pair','BS','Sector','NumBeams','Beamwidth_deg','PcorrectBeam','Overhead_percent'});
end

function T = make_combined_beam_table(NSGAK, MOWOAK, P)
Pair = (1:P.numPairs)';
BS = zeros(P.numPairs,1);
Sector = zeros(P.numPairs,1);
for q = 1:P.numPairs
    [BS(q), Sector(q)] = pair_to_bssec(q, P);
end
Fixed_8_Beams = P.baselineBeams*ones(P.numPairs,1);
Fixed_24_Beams = 24*ones(P.numPairs,1);
NSGA3_Beams = NSGAK(:);
MOWOA_Beams = MOWOAK(:);
NSGA3_Beamwidth_deg = P.sectorSpan_deg ./ NSGA3_Beams;
MOWOA_Beamwidth_deg = P.sectorSpan_deg ./ MOWOA_Beams;
NSGA3_PcorrectBeam = beam_correct_probability(NSGA3_Beams, P);
MOWOA_PcorrectBeam = beam_correct_probability(MOWOA_Beams, P);
Fixed_8_Overhead_percent = 100*compute_sector_overheads(P.baselineBeams*ones(1,P.numPairs), P);
Fixed_24_Overhead_percent = 100*compute_sector_overheads(24*ones(1,P.numPairs), P);
NSGA3_Overhead_percent = 100*compute_sector_overheads(NSGAK, P);
MOWOA_Overhead_percent = 100*compute_sector_overheads(MOWOAK, P);
T = table(Pair, BS, Sector, Fixed_8_Beams, Fixed_24_Beams, NSGA3_Beams, MOWOA_Beams, ...
    NSGA3_Beamwidth_deg, MOWOA_Beamwidth_deg, ...
    NSGA3_PcorrectBeam, MOWOA_PcorrectBeam, ...
    Fixed_8_Overhead_percent, Fixed_24_Overhead_percent, NSGA3_Overhead_percent, MOWOA_Overhead_percent);
end

function T = make_final_comparison_table(A, C, B, D)
Metric = {'Avg. SINR (dB)'; ...
          '5%-tile SINR (dB)'; ...
          'Coverage (SINR > 3 dB) (%)'; ...
          'Avg. Effective Throughput (Mbps/user)'; ...
          '5%-tile Throughput (Mbps/user)'; ...
          'Median Throughput (Mbps/user)'; ...
          'Jain Fairness Index of Throughput'; ...
          'Avg. Overhead (%)'; ...
          'Avg. Interference (dBm)'};
Fixed_8_Beams = metric_vector(A);
Fixed_24_Beams = metric_vector(C);
NSGA3_FoM_Best = metric_vector(B);
MOWOA_FoM_Best = metric_vector(D);
T = table(Metric, Fixed_8_Beams, Fixed_24_Beams, NSGA3_FoM_Best, MOWOA_FoM_Best);
end

function v = metric_vector(M)
v = [M.AvgSINR_dB; M.P5SINR_dB; M.Coverage_pct; M.AvgEffTput_Mbps; ...
     M.P5Tput_Mbps; M.MedianTput_Mbps; M.JFI; M.AvgOverhead_percent; ...
     M.AvgInterference_dBm];
end

function T = make_improvement_table(A, B, D)
Metric = {'Avg. Effective Throughput'; '5%-tile SINR'; 'JFI'; 'Avg. Overhead'; 'Avg. Interference'};
Unit = {'Mbps/user'; 'dB'; 'unitless'; 'percent'; 'dBm'};
Fixed_8_Beams = [A.AvgEffTput_Mbps; A.P5SINR_dB; A.JFI; A.AvgOverhead_percent; A.AvgInterference_dBm];
NSGA3_FoM_Best = [B.AvgEffTput_Mbps; B.P5SINR_dB; B.JFI; B.AvgOverhead_percent; B.AvgInterference_dBm];
MOWOA_FoM_Best = [D.AvgEffTput_Mbps; D.P5SINR_dB; D.JFI; D.AvgOverhead_percent; D.AvgInterference_dBm];

NSGA3_AbsoluteChange = NSGA3_FoM_Best - Fixed_8_Beams;
MOWOA_AbsoluteChange = MOWOA_FoM_Best - Fixed_8_Beams;

NSGA3_RelativeChange_percent = NaN(numel(Metric),1);
MOWOA_RelativeChange_percent = NaN(numel(Metric),1);
linearRows = [1 3 4];
for k = 1:numel(linearRows)
    r = linearRows(k);
    NSGA3_RelativeChange_percent(r) = 100*NSGA3_AbsoluteChange(r)/max(abs(Fixed_8_Beams(r)), eps);
    MOWOA_RelativeChange_percent(r) = 100*MOWOA_AbsoluteChange(r)/max(abs(Fixed_8_Beams(r)), eps);
end

PreferredDirection = {'Higher is better'; 'Higher is better'; 'Higher is better'; 'Lower is better'; 'Lower is better'};
T = table(Metric, Unit, Fixed_8_Beams, NSGA3_FoM_Best, MOWOA_FoM_Best, ...
    NSGA3_AbsoluteChange, MOWOA_AbsoluteChange, ...
    NSGA3_RelativeChange_percent, MOWOA_RelativeChange_percent, PreferredDirection);
end

function T = make_optimization_setup_table(P, NSGA, MOWOA)
Algorithm = {'NSGA-III'; 'MOWOA'};
Population_or_Whales = [P.popSize; P.popSize];
Generations_or_Iterations = [P.maxGen; P.maxGen];
MutationProbability = [P.pMutation; P.pMutation];
CrossoverProbability = [P.pCrossover; NaN];
WorkingArchiveLimit = [P.workingArchiveSize; P.workingArchiveSize];
FinalWorkingArchiveSize = [size(NSGA.workingArchive.population,1); ...
                           size(MOWOA.workingArchive.population,1)];
HistoricalArchiveLimit = {'Unbounded'; 'Unbounded'};
FinalHistoricalArchiveSize = [size(NSGA.historicalArchive.population,1); ...
                              size(MOWOA.historicalArchive.population,1)];
ApproxObjectiveEvaluations = [NSGA.approximateEvaluations; MOWOA.approximateEvaluations];
NativeSearchMechanism = {'Tournament + crossover + mutation + NSGA-III niching'; ...
                         'Encircling + random search + spiral + discrete mutation'};
T = table(Algorithm, Population_or_Whales, Generations_or_Iterations, ...
    MutationProbability, CrossoverProbability, WorkingArchiveLimit, ...
    FinalWorkingArchiveSize, HistoricalArchiveLimit, FinalHistoricalArchiveSize, ...
    ApproxObjectiveEvaluations, NativeSearchMechanism);
end

% =========================================================================
% Plotting functions
% =========================================================================
function plot_optimizer_history(H, optimizerName, prefix, outDir)
figure('Color','w');
plot(H.BestHistoricalArchiveFoM, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Generation / iteration');
ylabel('Best equal-weight FoM in historical archive');
title(sprintf('%s convergence according to historical archive FoM', optimizerName));
save_png_compat(gcf, fullfile(outDir, sprintf('fig_%s_fom_convergence.png', prefix)));
close(gcf);

figure('Color','w');
plot(H.BestAvgTput_Mbps, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Generation / iteration');
ylabel('Best average effective throughput (Mbps/user)');
title(sprintf('Best throughput observed during %s', optimizerName));
save_png_compat(gcf, fullfile(outDir, sprintf('fig_%s_best_throughput.png', prefix)));
close(gcf);

figure('Color','w');
plot(H.ParetoCount, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Generation / iteration');
ylabel('Number of rank-1 solutions in current population');
title(sprintf('Current Pareto-front size during %s', optimizerName));
save_png_compat(gcf, fullfile(outDir, sprintf('fig_%s_pareto_count.png', prefix)));
close(gcf);
end

function plot_history_comparison(H1, H2, outDir)
figure('Color','w');
plot(H1.BestHistoricalArchiveFoM, 'LineWidth', 1.6, 'DisplayName', 'NSGA-III');
hold on;
plot(H2.BestHistoricalArchiveFoM, 'LineWidth', 1.6, 'DisplayName', 'MOWOA');
grid on;
xlabel('Generation / iteration');
ylabel('Best historical-archive FoM (normalized within each archive)');
title('Historical-archive convergence: NSGA-III versus MOWOA');
legend('show','Location','best');
save_png_compat(gcf, fullfile(outDir, 'fig_convergence_nsga3_vs_mowoa.png'));
close(gcf);
end

function plot_best_beams_comparison(T, outDir)
figure('Color','w');
values = [T.Fixed_8_Beams, T.Fixed_24_Beams, T.NSGA3_Beams, T.MOWOA_Beams];
bar(T.Pair, values, 'grouped');
grid on;
xlabel('Sector pair index');
ylabel('Number of beams');
title('Beam count per sector: Fixed 8, Fixed 24, NSGA-III, and MOWOA');
set(gca, 'XTick', T.Pair);
legend({'Fixed N=8','Fixed N=24','NSGA-III','MOWOA'}, 'Location','best');
save_png_compat(gcf, fullfile(outDir, 'Fig2_Beam_Count_Per_Sector.png'));
close(gcf);
end

function plot_overhead_per_sector_comparison(T, outDir)
figure('Color','w');
values = [T.Fixed_8_Overhead_percent, T.NSGA3_Overhead_percent, T.MOWOA_Overhead_percent];
bar(T.Pair, values, 'grouped');
grid on;
xlabel('Sector pair index');
ylabel('Overhead (%)');
title('Overhead per sector: Fixed 8, NSGA-III, and MOWOA');
set(gca, 'XTick', T.Pair);
legend({'Fixed N=8','NSGA-III','MOWOA'}, 'Location','best');
save_png_compat(gcf, fullfile(outDir, 'fig_overhead_per_sector_comparison.png'));
close(gcf);
end

function plot_beam_petals(BestBeamTable, P, Dplot, outDir, algorithmLabel, fileName)
fig = figure('Color','w', 'Position', [100 40 720 720]);
hold on;
axis equal;
box on;
grid on;

sectorColors = [1.00 0.66 0.66; ...
                0.63 0.88 0.66; ...
                0.65 0.68 1.00];
petalLength = P.sectorRadius_m;

for q = 1:P.numPairs
    bs = BestBeamTable.BS(q);
    sec = BestBeamTable.Sector(q);
    K = BestBeamTable.NumBeams(q);
    HPBW = BestBeamTable.Beamwidth_deg(q);
    xb = P.bsPos(bs,1);
    yb = P.bsPos(bs,2);
    sectorStart = (sec-1) * P.sectorSpan_deg;

    for k = 1:K
        beamCenter = sectorStart + (k-0.5)*HPBW;
        theta = linspace(beamCenter - HPBW/2, beamCenter + HPBW/2, 60);
        radialShape = max(0, cosd(180*(theta - beamCenter)/HPBW));
        rr = petalLength * radialShape;
        xx = [xb, xb + rr.*cosd(theta), xb];
        yy = [yb, yb + rr.*sind(theta), yb];
        patch(xx, yy, sectorColors(sec,:), ...
            'FaceAlpha', 0.58, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

for bs = 1:P.numBS
    xb = P.bsPos(bs,1);
    yb = P.bsPos(bs,2);
    for ang = 0:P.sectorSpan_deg:(360-P.sectorSpan_deg)
        xe = xb + P.sectorRadius_m*cosd(ang);
        ye = yb + P.sectorRadius_m*sind(ang);
        plot([xb xe], [yb ye], '-', 'Color', [0.45 0.45 0.45], ...
            'LineWidth', 0.55, 'HandleVisibility', 'off');
    end
end

if nargin >= 3 && ~isempty(Dplot) && isfield(Dplot, 'x_m') && isfield(Dplot, 'y_m')
    plot(Dplot.x_m, Dplot.y_m, 'o', 'LineStyle', 'none', ...
        'MarkerSize', 3.1, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
        'LineWidth', 0.55, 'HandleVisibility', 'off');
end

for bs = 1:P.numBS
    xb = P.bsPos(bs,1);
    yb = P.bsPos(bs,2);
    plot(xb, yb, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4.5, ...
        'HandleVisibility', 'off');
    text(xb, yb+4, sprintf('BS%d', bs), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'FontWeight', 'bold', ...
        'FontSize', 10, 'Color', 'k');
end

xlim([0 1000]);
ylim([0 1000]);
set(gca, 'XTick', 0:100:1000, 'YTick', 0:100:1000, ...
    'FontName', 'Arial', 'FontSize', 9, 'LineWidth', 0.8, 'Layer', 'top');
title({'5G [Codebook-Beamforming] (7 BS \times 3 sectors)', ...
       ['Each 120^\circ sector split into ' algorithmLabel ' selected curved beams']}, ...
       'FontWeight', 'bold', 'FontSize', 11);

save_png_compat(fig, fullfile(outDir, fileName));
close(fig);
end

function plot_combined_beam_petals(NSGABestBeamTable, MOWOABestBeamTable, P, Dplot, outDir)
% PLOT_COMBINED_BEAM_PETALS
% Creates one publication-style figure containing the NSGA-III and MOWOA
% network-wide beam-pattern maps as two side-by-side panels.
fig = figure('Color','w', 'Position', [60 80 1180 560]);

subplot(1,2,1);
draw_beam_petals_panel(NSGABestBeamTable, P, Dplot);

subplot(1,2,2);
draw_beam_petals_panel(MOWOABestBeamTable, P, Dplot);

% Subfigure labels placed below the two panels, matching the manuscript style.
annotation(fig, 'textbox', [0.18 0.005 0.22 0.045], ...
    'String', '(a):  NSGA-III', 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
annotation(fig, 'textbox', [0.60 0.005 0.22 0.045], ...
    'String', '(b):  MOWOA', 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);

save_png_compat(fig, fullfile(outDir, 'Fig3_Network_Wide_Beam_Patterns.png'));
close(fig);
end

function draw_beam_petals_panel(BestBeamTable, P, Dplot)
% Draws one beam-pattern panel on the current axes without creating/saving
% a separate figure. The beam geometry is identical to the original code.
hold on;
axis equal;
box on;
grid on;

sectorColors = [1.00 0.66 0.66; ...
                0.63 0.88 0.66; ...
                0.65 0.68 1.00];
petalLength = P.sectorRadius_m;

for q = 1:P.numPairs
    bs = BestBeamTable.BS(q);
    sec = BestBeamTable.Sector(q);
    K = BestBeamTable.NumBeams(q);
    HPBW = BestBeamTable.Beamwidth_deg(q);
    xb = P.bsPos(bs,1);
    yb = P.bsPos(bs,2);
    sectorStart = (sec-1) * P.sectorSpan_deg;

    for k = 1:K
        beamCenter = sectorStart + (k-0.5)*HPBW;
        theta = linspace(beamCenter - HPBW/2, beamCenter + HPBW/2, 60);
        radialShape = max(0, cosd(180*(theta - beamCenter)/HPBW));
        rr = petalLength * radialShape;
        xx = [xb, xb + rr.*cosd(theta), xb];
        yy = [yb, yb + rr.*sind(theta), yb];
        patch(xx, yy, sectorColors(sec,:), ...
            'FaceAlpha', 0.58, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

for bs = 1:P.numBS
    xb = P.bsPos(bs,1);
    yb = P.bsPos(bs,2);
    for ang = 0:P.sectorSpan_deg:(360-P.sectorSpan_deg)
        xe = xb + P.sectorRadius_m*cosd(ang);
        ye = yb + P.sectorRadius_m*sind(ang);
        plot([xb xe], [yb ye], '-', 'Color', [0.45 0.45 0.45], ...
            'LineWidth', 0.55, 'HandleVisibility', 'off');
    end
end

if nargin >= 3 && ~isempty(Dplot) && isfield(Dplot, 'x_m') && isfield(Dplot, 'y_m')
    plot(Dplot.x_m, Dplot.y_m, 'o', 'LineStyle', 'none', ...
        'MarkerSize', 2.8, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
        'LineWidth', 0.50, 'HandleVisibility', 'off');
end

for bs = 1:P.numBS
    xb = P.bsPos(bs,1);
    yb = P.bsPos(bs,2);
    plot(xb, yb, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4.0, ...
        'HandleVisibility', 'off');
    text(xb, yb+4, sprintf('BS%d', bs), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', 'FontWeight', 'bold', ...
        'FontSize', 8, 'Color', 'k');
end

xlim([0 1000]);
ylim([0 1000]);
set(gca, 'XTick', 0:100:1000, 'YTick', 0:100:1000, ...
    'FontName', 'Arial', 'FontSize', 7.5, 'LineWidth', 0.8, 'Layer', 'top');
end

function plot_combined_cdf_figure(BaselineRows, Fixed24Rows, NSGABestRows, MOWOABestRows, outDir)
% PLOT_COMBINED_CDF_FIGURE
% Creates one publication-style figure with the SINR CDF and effective-
% throughput CDF as two side-by-side panels.
fig = figure('Color','w', 'Position', [60 120 1180 470]);

subplot(1,2,1);
draw_cdf4_panel(BaselineRows.sinr_dB, Fixed24Rows.sinr_dB, NSGABestRows.sinr_dB, MOWOABestRows.sinr_dB, ...
    'N = 8 beams', 'N = 24 beams', 'NSGA-III selected beams', 'MOWOA selected beams', ...
    'Downlink SINR (dB)', 'CDF of Downlink SINR: Fixed 8 vs Fixed 24 vs NSGA-III vs MOWOA');

subplot(1,2,2);
draw_cdf4_panel(BaselineRows.tput_Mbps, Fixed24Rows.tput_Mbps, NSGABestRows.tput_Mbps, MOWOABestRows.tput_Mbps, ...
    'N = 8 beams', 'N = 24 beams', 'NSGA-III selected beams', 'MOWOA selected beams', ...
    'Effective throughput (Mbps/user)', ...
    'CDF of Effective Throughput: Fixed 8 vs Fixed 24 vs NSGA-III vs MOWOA');

annotation(fig, 'textbox', [0.18 0.005 0.22 0.045], ...
    'String', '(a):  SINR CDF', 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);
annotation(fig, 'textbox', [0.60 0.005 0.22 0.045], ...
    'String', '(b):  Effective-throughput CDF', 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10);

save_png_compat(fig, fullfile(outDir, 'Fig4_CDF_SINR_Throughput.png'));
close(fig);
end

function draw_cdf4_panel(x1, x2, x3, x4, l1, l2, l3, l4, xlab, ttl)
% Draws one four-curve empirical CDF on the current axes.
% A magnified inset is added for the dashed rectangular region so that
% small differences among the four curves are easier to distinguish.
mainAx = gca;
fig = ancestor(mainAx, 'figure');

x1 = sort(x1(:));
x2 = sort(x2(:));
x3 = sort(x3(:));
x4 = sort(x4(:));
y1 = (1:numel(x1))'/numel(x1);
y2 = (1:numel(x2))'/numel(x2);
y3 = (1:numel(x3))'/numel(x3);
y4 = (1:numel(x4))'/numel(x4);

h1 = plot(x1, y1, 'LineWidth', 1.5, 'DisplayName', l1);
hold on;
h2 = plot(x2, y2, 'LineWidth', 1.5, 'DisplayName', l2);
h3 = plot(x3, y3, 'LineWidth', 1.5, 'DisplayName', l3);
h4 = plot(x4, y4, 'LineWidth', 1.5, 'DisplayName', l4);
grid on;
box on;
xlabel(xlab);
ylabel('CDF');
title(ttl, 'FontSize', 9, 'FontWeight', 'bold');
legend('show','Location','best');
ylim([0 1]);
set(gca, 'FontSize', 8.5);

% ---------------------------------------------------------------------
% Magnified region: use the same central CDF interval indicated by the
% dashed rectangle in the main plot.
% ---------------------------------------------------------------------
zoomYMin = 0.35;
zoomYMax = 0.65;

i1a = max(1, min(numel(x1), round(zoomYMin*numel(x1))));
i1b = max(1, min(numel(x1), round(zoomYMax*numel(x1))));
i2a = max(1, min(numel(x2), round(zoomYMin*numel(x2))));
i2b = max(1, min(numel(x2), round(zoomYMax*numel(x2))));
i3a = max(1, min(numel(x3), round(zoomYMin*numel(x3))));
i3b = max(1, min(numel(x3), round(zoomYMax*numel(x3))));
i4a = max(1, min(numel(x4), round(zoomYMin*numel(x4))));
i4b = max(1, min(numel(x4), round(zoomYMax*numel(x4))));

xZoomMin = min([x1(i1a), x2(i2a), x3(i3a), x4(i4a)]);
xZoomMax = max([x1(i1b), x2(i2b), x3(i3b), x4(i4b)]);

xSpan = xZoomMax - xZoomMin;
if xSpan < eps
    xSpan = max(abs(xZoomMax), 1) * 0.05;
end
xPad = 0.10 * xSpan;
xZoomMin = xZoomMin - xPad;
xZoomMax = xZoomMax + xPad;

% Dashed rectangle that identifies the region being magnified.
rectangle('Parent', mainAx, ...
    'Position', [xZoomMin, zoomYMin, xZoomMax-xZoomMin, zoomYMax-zoomYMin], ...
    'LineStyle', '--', 'LineWidth', 0.9);

% ---------------------------------------------------------------------
% Place the magnified inset on the side opposite the dashed rectangle.
% This avoids covering the region of interest and gives a magnifying-glass
% appearance when combined with the connector lines below.
% ---------------------------------------------------------------------
mainPos = get(mainAx, 'Position');
mainXLim = get(mainAx, 'XLim');
zoomCenter = 0.5*(xZoomMin + xZoomMax);
zoomRelativeX = (zoomCenter - mainXLim(1)) / max(mainXLim(2)-mainXLim(1), eps);

if zoomRelativeX >= 0.5
    % Dashed region is toward the right: put inset at lower left.
    insetPos = [mainPos(1) + 0.06*mainPos(3), ...
                mainPos(2) + 0.10*mainPos(4), ...
                0.34*mainPos(3), ...
                0.32*mainPos(4)];
    connectFromX = xZoomMin;
    insetConnectX = insetPos(1) + insetPos(3);
else
    % Dashed region is toward the left: put inset at lower right.
    insetPos = [mainPos(1) + 0.60*mainPos(3), ...
                mainPos(2) + 0.10*mainPos(4), ...
                0.34*mainPos(3), ...
                0.32*mainPos(4)];
    connectFromX = xZoomMax;
    insetConnectX = insetPos(1);
end

insetAx = axes('Parent', fig, 'Position', insetPos, ...
    'Box', 'on', 'LineWidth', 1.1, 'Color', 'w');
plot(insetAx, x1, y1, 'LineWidth', 1.25, 'Color', get(h1, 'Color'));
hold(insetAx, 'on');
plot(insetAx, x2, y2, 'LineWidth', 1.25, 'Color', get(h2, 'Color'));
plot(insetAx, x3, y3, 'LineWidth', 1.25, 'Color', get(h3, 'Color'));
plot(insetAx, x4, y4, 'LineWidth', 1.25, 'Color', get(h4, 'Color'));
grid(insetAx, 'on');
box(insetAx, 'on');
xlim(insetAx, [xZoomMin xZoomMax]);
ylim(insetAx, [zoomYMin zoomYMax]);
title(insetAx, 'Magnified view', 'FontSize', 7.5, 'FontWeight', 'bold');
set(insetAx, 'FontSize', 7, 'Layer', 'top');

% ---------------------------------------------------------------------
% Connector lines from the dashed rectangle to the magnified inset.
% Convert the two selected rectangle corners from data coordinates to
% normalized figure coordinates for annotation().
% ---------------------------------------------------------------------
mainYLim = get(mainAx, 'YLim');
xNorm = mainPos(1) + mainPos(3) * ...
    (connectFromX - mainXLim(1)) / max(mainXLim(2)-mainXLim(1), eps);
yNormLow = mainPos(2) + mainPos(4) * ...
    (zoomYMin - mainYLim(1)) / max(mainYLim(2)-mainYLim(1), eps);
yNormHigh = mainPos(2) + mainPos(4) * ...
    (zoomYMax - mainYLim(1)) / max(mainYLim(2)-mainYLim(1), eps);

annotation(fig, 'line', [xNorm insetConnectX], ...
    [yNormLow insetPos(2)], 'LineWidth', 0.8);
annotation(fig, 'line', [xNorm insetConnectX], ...
    [yNormHigh insetPos(2)+insetPos(4)], 'LineWidth', 0.8);

% Do not reactivate mainAx here; keeping insetAx on top makes the magnified
% view visible in the final saved figure.
end

function plot_pareto_scatter(metrics, FoM, optimizerName, prefix, outDir)
figure('Color','w');
scatter3(metrics(:,1), metrics(:,2), metrics(:,4), 55, FoM, 'filled');
grid on;
xlabel('Avg. effective throughput (Mbps/user)');
ylabel('5%-tile SINR (dB)');
zlabel('Avg. overhead (%)');
title(sprintf('%s Pareto front: throughput, 5%%-tile SINR, and overhead', optimizerName));
cb = colorbar;
ylabel(cb, 'Equal-weight FoM');
save_png_compat(gcf, fullfile(outDir, sprintf('fig_%s_pareto_tput_p5_overhead.png', prefix)));
close(gcf);

figure('Color','w');
scatter(metrics(:,3), metrics(:,5), 55, FoM, 'filled');
grid on;
xlabel('Jain fairness index');
ylabel('Avg. interference (dBm)');
title(sprintf('%s Pareto front: fairness versus interference', optimizerName));
cb = colorbar;
ylabel(cb, 'Equal-weight FoM');
save_png_compat(gcf, fullfile(outDir, sprintf('fig_%s_pareto_fairness_interference.png', prefix)));
close(gcf);
end

function plot_combined_pareto_scatter(M1, M2, outDir)
figure('Color','w');
h1 = scatter3(M1(:,1), M1(:,2), M1(:,4), 50, 'filled');
set(h1, 'Marker', 'o', 'DisplayName', 'NSGA-III');
hold on;
h2 = scatter3(M2(:,1), M2(:,2), M2(:,4), 50, 'filled');
set(h2, 'Marker', '^', 'DisplayName', 'MOWOA');
grid on;
xlabel('Avg. effective throughput (Mbps/user)');
ylabel('5%-tile SINR (dB)');
zlabel('Avg. overhead (%)');
title('Pareto archive comparison: NSGA-III versus MOWOA');
legend('show','Location','best');
save_png_compat(gcf, fullfile(outDir, 'Fig1_Pareto_Archive_Comparison.png'));
close(gcf);
end

function plot_hist3(x1, x2, x3, l1, l2, l3, xlab, ttl, fname)
figure('Color','w');
mn = min([x1(:); x2(:); x3(:)]);
mx = max([x1(:); x2(:); x3(:)]);
if mx <= mn
    mx = mn + 1;
end
edges = linspace(mn, mx, 41);
histogram(x1, edges, 'FaceAlpha', 0.45, 'DisplayName', l1);
hold on;
histogram(x2, edges, 'FaceAlpha', 0.45, 'DisplayName', l2);
histogram(x3, edges, 'FaceAlpha', 0.45, 'DisplayName', l3);
grid on;
xlabel(xlab);
ylabel('Users');
title(ttl);
legend('show','Location','best');
save_png_compat(gcf, fname);
close(gcf);
end

function plot_cdf3(x1, x2, x3, l1, l2, l3, xlab, ttl, fname)
figure('Color','w');
x1 = sort(x1(:)); x2 = sort(x2(:)); x3 = sort(x3(:));
y1 = (1:numel(x1))'/numel(x1);
y2 = (1:numel(x2))'/numel(x2);
y3 = (1:numel(x3))'/numel(x3);
plot(x1, y1, 'LineWidth', 1.5, 'DisplayName', l1);
hold on;
plot(x2, y2, 'LineWidth', 1.5, 'DisplayName', l2);
plot(x3, y3, 'LineWidth', 1.5, 'DisplayName', l3);
grid on;
xlabel(xlab);
ylabel('CDF');
title(ttl);
legend('show','Location','best');
save_png_compat(gcf, fname);
close(gcf);
end

function plot_metric_bar3(valueFixed8, valueNSGA3, valueMOWOA, ylab, ttl, fname, yLimits)
figure('Color','w');
values = [valueFixed8, valueNSGA3, valueMOWOA];
bar(1:3, values, 0.62);
grid on;
box on;
set(gca, 'XTick', 1:3, ...
         'XTickLabel', {'N = 8 beams', 'NSGA-III', 'MOWOA'}, ...
         'FontSize', 10);
ylabel(ylab);
title(ttl);

if nargin >= 7 && ~isempty(yLimits)
    ylim(yLimits);
else
    vmin = min(values);
    vmax = max(values);
    span = vmax - vmin;
    if span < 1e-12
        span = max(abs(vmax), 1);
    end
    lowerPad = 0.15 * span;
    upperPad = 0.25 * span;
    if vmin >= 0
        ylim([0, vmax + upperPad]);
    else
        ylim([vmin - lowerPad, vmax + upperPad]);
    end
end

yl = ylim;
yRange = yl(2) - yl(1);
for ii = 1:3
    yText = values(ii) + 0.025*yRange;
    text(ii, yText, sprintf('%.4f', values(ii)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold', 'FontSize', 10);
end

save_png_compat(gcf, fname);
close(gcf);
end

function save_png_compat(figHandle, fname)
set(figHandle, 'PaperPositionMode', 'auto');
print(figHandle, fname, '-dpng', '-r300');
end

% =========================================================================
% Utility functions
% =========================================================================
function [bs, sec] = pair_to_bssec(pair, P)
bs = floor((pair-1)/P.numSec) + 1;
sec = mod(pair-1, P.numSec) + 1;
end


function loadVec = generate_sector_loads(P)
if isfield(P, 'useNonUniformLoad') && P.useNonUniformLoad
    loadVec = P.sectorLoadProfile(:);
    jitter = round((2*rand(P.numPairs,1) - 1) * P.loadJitterUE);
    loadVec = loadVec + jitter;
    loadVec = max(P.minUEsPerSec, min(P.maxUEsPerSec, loadVec));
else
    loadVec = P.UEsPerSec * ones(P.numPairs,1);
end
loadVec = round(loadVec(:));
end

function pCorr = beam_correct_probability(K, P)
% K-dependent beam-selection probability. The reference is pCorrectAt8 at
% K=8. Larger K means narrower beams and a higher probability of beam-index
% error; smaller K means wider beams and a more robust selection.
pCorr = P.pCorrectAt8 - P.pCorrectSlope * (K - 8);
pCorr = max(P.pCorrectMin, min(P.pCorrectMax, pCorr));
end

function PLdB = path_loss_uma_simple(d_m, f_Hz)
% Close, simple UMa-like path-loss expression retained from the original code.
% If your manuscript states the full 3GPP TR 38.901 UMa LOS/NLOS model,
% replace this function with the exact TR 38.901 implementation.
f_GHz = f_Hz/1e9;
d_m = max(d_m, 1);
PLdB = 32.4 + 21*log10(d_m) + 20*log10(f_GHz);
end

function G = beam_gain(delta, HPBW, Gmax, Am)
G = Gmax - min(12*(delta./HPBW).^2, Am);
end

function ang = wrap180_local(ang)
ang = mod(ang + 180, 360) - 180;
end

function J = jain_fairness(x)
x = x(:);
x(~isfinite(x)) = 0;
den = numel(x)*sum(x.^2);
if den <= 0
    J = 0;
else
    J = (sum(x)^2)/den;
end
end

function write_table_word(T, fname, tableTitle)
% WRITE_TABLE_WORD
% Exports a MATLAB table T to a Word-readable document (fname, typically
% with a .doc extension). The file is written as HTML content that
% Microsoft Word opens natively and renders as a formatted table, so no
% extra MATLAB toolbox (e.g. Report Generator) is required.
try
    fid = fopen(fname, 'w');
    if fid == -1
        warning('write_table_word: could not open file for writing: %s', fname);
        return;
    end
    fprintf(fid, '<html xmlns:o="urn:schemas-microsoft-com:office:office" ');
    fprintf(fid, 'xmlns:w="urn:schemas-microsoft-com:office:word" ');
    fprintf(fid, 'xmlns="http://www.w3.org/TR/REC-html40">\n');
    fprintf(fid, '<head><meta charset="UTF-8"><title>%s</title>\n', tableTitle);
    fprintf(fid, ['<style> body{font-family:Calibri,Arial,sans-serif;} ' ...
        'table{border-collapse:collapse;width:100%%;} ' ...
        'th,td{border:1px solid #444444;padding:6px 10px;text-align:center;font-size:11pt;} ' ...
        'th{background-color:#D9E1F2;font-weight:bold;} ' ...
        'h1{font-size:14pt;color:#1F4E78;font-family:Calibri,Arial,sans-serif;} </style>\n']);
    fprintf(fid, '</head><body>\n');
    fprintf(fid, '<h1>%s</h1>\n', tableTitle);
    fprintf(fid, '<table>\n<tr>');

    varNames = T.Properties.VariableNames;
    for c = 1:numel(varNames)
        fprintf(fid, '<th>%s</th>', strrep(varNames{c}, '_', ' '));
    end
    fprintf(fid, '</tr>\n');

    nRows = height(T);
    for r = 1:nRows
        fprintf(fid, '<tr>');
        for c = 1:numel(varNames)
            val = T{r, c};
            if iscell(val)
                val = val{1};
            end
            if isnumeric(val)
                if isscalar(val)
                    if isnan(val)
                        cellStr = 'NaN';
                    elseif val == round(val)
                        cellStr = sprintf('%d', val);
                    else
                        cellStr = sprintf('%.4f', val);
                    end
                else
                    cellStr = mat2str(val);
                end
            elseif ischar(val)
                cellStr = val;
            elseif isstring(val)
                cellStr = char(val);
            else
                cellStr = char(string(val));
            end
            fprintf(fid, '<td>%s</td>', cellStr);
        end
        fprintf(fid, '</tr>\n');
    end

    fprintf(fid, '</table>\n</body></html>');
    fclose(fid);
catch ME
    warning('write_table_word: failed to write %s (%s)', fname, ME.message);
end
end
