function kwave_finite(input_json_path, output_mat_path, case_name)
%KWAVE_FINITE Propagate a prescribed, symmetric thermal monopole sheet.
% The source phasor uses exp(+i*omega*t), with the original LBM clock.
% This is a homogeneous lossless propagation model, not a membrane solver.

arguments
    input_json_path (1,:) char
    output_mat_path (1,:) char
    case_name (1,:) char
end

assert(isfile(input_json_path), 'Input JSON does not exist.');
assert(~isfile(output_mat_path) && ~isfolder(output_mat_path), ...
    'Refusing to overwrite an existing output path.');
output_dir = fileparts(output_mat_path);
assert(isempty(output_dir) || isfolder(output_dir), ...
    'The caller must create the output directory.');
cfg = jsondecode(fileread(input_json_path));
assert(isfolder(cfg.toolbox_path), 'The configured k-Wave path does not exist.');
old_path = path;
path_cleanup = onCleanup(@() path(old_path));
addpath(cfg.toolbox_path, '-begin');
assert(strcmpi(strrep(which('acousticFieldPropagator'), '\', '/'), ...
    strrep(fullfile(cfg.toolbox_path, 'acousticFieldPropagator.m'), '\', '/')), ...
    'The configured acousticFieldPropagator must be the active implementation.');

case_index = find(strcmp({cfg.cases.name}, case_name));
assert(isscalar(case_index), 'Case name must identify exactly one configured case.');
run_cfg = cfg.cases(case_index);
assert(ismember(case_name, ...
    {'plane_check', 'finite_coarse', 'finite_fine', 'finite_time', 'finite_padding'}), ...
    'Unsupported case name.');
frequency_Hz = cfg.frequency_Hz;
c0_m_s = cfg.c0_m_s;
rho0_kg_m3 = cfg.rho0_kg_m3;
dx_m = run_cfg.dx_m;
evaluation_time_s = run_cfg.evaluation_cycles / frequency_Hz;
grid_expansion_factor = run_cfg.grid_expansion_factor;
positive_values = [frequency_Hz, c0_m_s, rho0_kg_m3, dx_m, ...
    evaluation_time_s, cfg.width_m, cfg.height_m];
assert(all(isfinite(positive_values) & positive_values > 0), ...
    'Frequency, material properties, dimensions and evaluation time must be positive.');
assert(isfinite(grid_expansion_factor) && grid_expansion_factor >= 1, ...
    'Grid expansion factor must be at least one.');
source_pressure_Pa = complex(cfg.source_re, cfg.source_im);
assert(isscalar(source_pressure_Pa) && isfinite(source_pressure_Pa) ...
    && abs(source_pressure_Pa) > 0, 'Source phasor must be finite and nonzero.');

z_m = symmetric_axis(0.045, dx_m);
iz_source = (numel(z_m) + 1) / 2;
if strcmp(case_name, 'plane_check')
    x_m = 0;
    y_m = 0;
    source_weights = zeros(numel(z_m), 1);
    source_weights(iz_source) = 1;
    amp_in = abs(source_pressure_Pa) * source_weights;
    source_area_m2 = 0; % Infinite planar normalization check; no finite area.
else
    x_m = symmetric_axis(0.030, dx_m);
    y_m = symmetric_axis(0.015, dx_m);
    assert(cfg.width_m <= 0.060 && cfg.height_m <= 0.030, ...
        'The configured sheet does not fit the output grid.');
    wx = cell_overlap(x_m, cfg.width_m, dx_m);
    wy = cell_overlap(y_m, cfg.height_m, dx_m);
    source_weights = wx * wy.';
    source_area_m2 = sum(source_weights, 'all') * dx_m^2;
    expected_area = cfg.width_m * cfg.height_m;
    assert(abs(source_area_m2 - expected_area) <= 1e-12 * expected_area, ...
        'The discretized source area does not equal the physical sheet area.');
    amp_in = zeros(numel(x_m), numel(y_m), numel(z_m));
    amp_in(:, :, iz_source) = abs(source_pressure_Pa) * source_weights;
end

% The complete output region must be reached after the one-cycle ramp.
max_distance_m = sqrt((max(abs(x_m))+cfg.width_m/2)^2 ...
    + (max(abs(y_m))+cfg.height_m/2)^2 + max(abs(z_m))^2);
assert(evaluation_time_s > max_distance_m/c0_m_s + 1/frequency_Hz, ...
    'Evaluation time is too early for the entire field and source ramp.');

% AFP already normalizes a grid-plane source to this pressure in EACH
% outgoing direction. Multiplying by two would double-count the two sides.
fprintf('Starting %s: dx = %.6g m, time = %.6g s, expansion = %.4g\n', ...
    case_name, dx_m, evaluation_time_s, grid_expansion_factor);
run_timer = tic;
pressure_at_time = acousticFieldPropagator(amp_in, angle(source_pressure_Pa), ...
    dx_m, frequency_Hz, c0_m_s, 'Time', evaluation_time_s, ...
    'GridExpansionFactor', grid_expansion_factor, 'UseRamp', true);
elapsed_s = toc(run_timer);
% AFP returns the complex field at Time; remove only its known clock factor.
pressure = pressure_at_time * exp(-1i * 2 * pi * frequency_Hz * evaluation_time_s);
assert(all(isfinite(pressure), 'all'), 'k-Wave returned a non-finite field.');
assert(isequal(size(pressure), size(amp_in)), 'k-Wave output shape mismatch.');
comp_info = getComputerInfo();
matlab_version = comp_info.matlab_version;
kwave_version = comp_info.kwave_version;
save(output_mat_path, 'x_m', 'y_m', 'z_m', 'pressure', 'source_weights', ...
    'source_area_m2', 'elapsed_s', 'matlab_version', 'kwave_version', ...
    'case_name', 'dx_m', 'evaluation_time_s', 'grid_expansion_factor', ...
    'frequency_Hz', 'c0_m_s', 'rho0_kg_m3', 'source_pressure_Pa', '-v7');
fprintf('Completed %s in %.3f s; saved %s\n', case_name, elapsed_s, output_mat_path);
end

function values = symmetric_axis(half_extent, dx)
n = round(half_extent / dx);
assert(abs(n * dx - half_extent) <= 1e-12 * half_extent, ...
    'Grid spacing must resolve the specified domain endpoints exactly.');
values = (-n:n).' * dx;
end

function weights = cell_overlap(centres, width, dx)
% Fraction of each node-centred cell covered by the physical rectangle.
weights = max(0, min(centres + dx / 2, width / 2) ...
    - max(centres - dx / 2, -width / 2)) / dx;
end
