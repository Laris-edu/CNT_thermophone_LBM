function kwave_handoff(input_mat_path, output_mat_path)
%KWAVE_HANDOFF Closed-curve p/u_n handoff to the unmodified k-Wave 2D solver.
% Inputs are SI phasors using Re(hat*exp(+i*omega*t)) and outward normals.
% The curve must enclose the source, in linear homogeneous acoustic gas.
% This adapter reproduces the exterior field and cancels the interior field.

arguments
    input_mat_path (1,:) char
    output_mat_path (1,:) char
end
assert(isfile(input_mat_path), 'Input MAT file does not exist.');
assert(~isfile(output_mat_path) && ~isfolder(output_mat_path), ...
    'Refusing to overwrite an existing output path.');
cfg = load(input_mat_path);
kwave_backend_requested = 'cpu';
if isfield(cfg, 'kwave_backend_requested')
    kwave_backend_requested = char(cfg.kwave_backend_requested);
end
assert(any(strcmp(kwave_backend_requested, {'cpu', 'matlab-gpu'})), ...
    'k-Wave backend must be cpu or matlab-gpu.');
kwave_backend_effective = kwave_backend_requested;
data_cast = 'double';
gpu_provenance = struct('enabled', false);
if strcmp(kwave_backend_requested, 'matlab-gpu')
    assert(license('test', 'Distrib_Computing_Toolbox'), ...
        'MATLAB Parallel Computing Toolbox is required for the matlab-gpu backend.');
    device = gpuDevice;
    gpu_provenance = struct('enabled', true, 'name', device.Name, ...
        'index', device.Index, 'compute_capability', device.ComputeCapability, ...
        'total_memory_bytes', device.TotalMemory, 'available_memory_bytes', device.AvailableMemory);
    for name = {'DriverVersion', 'ToolkitVersion'}
        if isprop(device, name{1})
            gpu_provenance.(name{1}) = device.(name{1});
        end
    end
    data_cast = 'gpuArray-double';
end
gpu_provenance_json = jsonencode(gpu_provenance);
scalar_fields = {'rho0_kg_m3', 'c0_m_s', 'frequency_Hz', 'dx_m', ...
    'domain_half_width_m', 'domain_half_height_m', 'evaluation_cycles', ...
    'ramp_cycles', 'record_cycles', 'cfl', 'pml_size'};
for index = 1:numel(scalar_fields)
    name = scalar_fields{index};
    validateattributes(cfg.(name), {'numeric'}, {'real', 'finite', 'positive', 'scalar'}, '', name);
    cfg.(name) = double(cfg.(name));
end
assert(isfield(cfg, 'source_validation_passed') ...
    && (isnumeric(cfg.source_validation_passed) || islogical(cfg.source_validation_passed)) ...
    && isreal(cfg.source_validation_passed) ...
    && isscalar(cfg.source_validation_passed) ...
    && isfinite(cfg.source_validation_passed) ...
    && any(cfg.source_validation_passed == [0, 1]), ...
    'source_validation_passed must be a finite scalar 0 or 1.');
source_validation_passed = logical(cfg.source_validation_passed);
diagnostic_mode = false;
if isfield(cfg, 'diagnostic_mode')
    assert((isnumeric(cfg.diagnostic_mode) || islogical(cfg.diagnostic_mode)) ...
        && isreal(cfg.diagnostic_mode) && isscalar(cfg.diagnostic_mode) ...
        && isfinite(cfg.diagnostic_mode) && any(cfg.diagnostic_mode == [0, 1]), ...
        'diagnostic_mode must be a finite scalar 0 or 1.');
    diagnostic_mode = logical(cfg.diagnostic_mode);
end
assert(source_validation_passed || diagnostic_mode, ...
    'The upstream source must pass validation before propagation.');
source_validation_bypassed = diagnostic_mode && ~source_validation_passed;
if source_validation_bypassed
    fprintf(['DIAGNOSTIC propagation of an UNVALIDATED source: source_validation_passed=0; ' ...
        'source_validation_bypassed=1. This result is not a validated source-to-sound chain.\n']);
end
old_path = path;
path_cleanup = onCleanup(@() path(old_path));
addpath(cfg.toolbox_path, '-begin');
assert(strcmpi(strrep(which('kspaceFirstOrder2D'), '\', '/'), ...
    strrep(fullfile(cfg.toolbox_path, 'kspaceFirstOrder2D.m'), '\', '/')), ...
    'The configured k-Wave installation is not active.');

sx = cfg.x_m(:); sy = cfg.y_m(:);
nx = cfg.normal_x(:); ny = cfg.normal_y(:); ds = cfg.ds_m(:);
ph = cfg.p_hat_Pa(:); un = cfg.un_hat_m_s(:);
assert(numel(sx) >= 4 && all([numel(sy), numel(nx), numel(ny), ...
    numel(ds), numel(ph), numel(un)] == numel(sx)), 'Curve arrays differ in size.');
assert(all(isfinite([sx; sy; nx; ny; ds; ph; un])) && all(ds > 0), ...
    'Curve data must be finite and arc lengths positive.');
assert(max(abs(hypot(nx, ny) - 1)) < 1e-10, 'Normals must be unit vectors.');
assert(norm([sum(nx .* ds), sum(ny .* ds)]) < 1e-8 * sum(ds), ...
    'Weighted normals do not close; an open top line is not a valid source.');
rho0_kg_m3 = cfg.rho0_kg_m3;
c0_m_s = cfg.c0_m_s;
frequency_Hz = cfg.frequency_Hz;
dx_m = cfg.dx_m;
required_positive = [rho0_kg_m3, c0_m_s, frequency_Hz, dx_m, ...
    cfg.domain_half_width_m, cfg.domain_half_height_m, cfg.evaluation_cycles, ...
    cfg.ramp_cycles, cfg.record_cycles, cfg.cfl, cfg.pml_size];
assert(all(isfinite(required_positive) & required_positive > 0), ...
    'Grid, material and time settings must be finite and positive.');
assert(cfg.cfl <= 0.3 && cfg.pml_size == round(cfg.pml_size), ...
    'Use CFL <= 0.3 and an integer PML size.');
assert(cfg.record_cycles == round(cfg.record_cycles) ...
    && cfg.evaluation_cycles > cfg.ramp_cycles + cfg.record_cycles, ...
    'Record whole periods after the ramp.');
Nx = 2 * ceil(cfg.domain_half_width_m / dx_m) + 1;
Ny = 2 * ceil(cfg.domain_half_height_m / dx_m) + 1;
kgrid = kWaveGrid(Nx, dx_m, Ny, dx_m);
x_grid_m = kgrid.x_vec;
y_grid_m = kgrid.y_vec;

% k-Wave's additive pressure input becomes dp/dt = 2*c/dx*source.p;
% its velocity input becomes du/dt = 2*c/dx*source.u. Thus the equivalent
% mass source rho*u_n*delta_Gamma and force p*n*delta_Gamma require HALVES,
% not copies of the measured pressure and velocity. ds/dx is quadrature.
impedance = rho0_kg_m3 * c0_m_s;
pressure_source_hat = deposit(sx, sy, impedance * un .* ds / (2 * dx_m), ...
    x_grid_m, y_grid_m, dx_m);
ux_source_hat = deposit(sx, sy, ph .* nx .* ds / (2 * impedance * dx_m), ...
    x_grid_m + dx_m / 2, y_grid_m, dx_m);
uy_source_hat = deposit(sx, sy, ph .* ny .* ds / (2 * impedance * dx_m), ...
    x_grid_m, y_grid_m + dx_m / 2, dx_m);
source.p_mask = pressure_source_hat ~= 0;
source.u_mask = (ux_source_hat ~= 0) | (uy_source_hat ~= 0);
source.p_mode = 'additive';
source.u_mode = 'additive';

% Resolve the source period exactly, then sample each rate at the midpoint
% of its leapfrog update: force at n*dt; mass at (n+1/2)*dt. Returned pressure
% after iteration n is at (n+1)*dt, not at kgrid.t_array(n+1).
points_per_cycle = ceil(c0_m_s / (frequency_Hz * dx_m * cfg.cfl));
dt_s = 1 / (frequency_Hz * points_per_cycle);
Nt = ceil(cfg.evaluation_cycles * points_per_cycle);
kgrid.setTime(Nt, dt_s);
omega = 2 * pi * frequency_Hz;
force_time = (0:Nt-1) * dt_s;
mass_time = force_time + dt_s / 2;
force_signal = harmonic(force_time, omega, cfg.ramp_cycles / frequency_Hz);
mass_signal = harmonic(mass_time, omega, cfg.ramp_cycles / frequency_Hz);
source.p = real(pressure_source_hat(source.p_mask) * mass_signal);
source.ux = real(ux_source_hat(source.u_mask) * force_signal);
source.uy = real(uy_source_hat(source.u_mask) * force_signal);
% Leave source_kappa to k-Wave's additive-source correction. No frequency
% override, gain fitting, or extra source correction is applied here.

requested_observer_x_m = cfg.observer_x_m(:);
requested_observer_y_m = cfg.observer_y_m(:);
assert(numel(requested_observer_x_m) == numel(requested_observer_y_m), ...
    'Observer coordinate arrays differ in size.');
ix = round((requested_observer_x_m - x_grid_m(1)) / dx_m) + 1;
iy = round((requested_observer_y_m - y_grid_m(1)) / dx_m) + 1;
assert(all(ix >= 1 & ix <= Nx & iy >= 1 & iy <= Ny), ...
    'An observer lies outside the physical grid.');
observer_x_m = x_grid_m(ix);
observer_y_m = y_grid_m(iy);
observer_indices = sub2ind([Nx, Ny], ix, iy);
sensor.mask = zeros(Nx, Ny);
sensor.mask(observer_indices) = 1;
sensor.record = {'p'};
record_steps = cfg.record_cycles * points_per_cycle;
sensor.record_start_index = Nt - record_steps + 1;
medium.sound_speed = c0_m_s;
medium.density = rho0_kg_m3;
fprintf('Closed-curve handoff: %d samples, %d x %d grid, %d steps\n', ...
    numel(ds), Nx, Ny, Nt);
run_timer = tic;
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
    'PMLInside', false, 'PMLSize', cfg.pml_size, 'PlotSim', false, ...
    'DataCast', data_cast, 'DataRecast', true, 'Smooth', false);
elapsed_s = toc(run_timer);
[found, ordering] = ismember(observer_indices, find(sensor.mask));
assert(all(found), 'Observer ordering was lost.');
observer_pressure_Pa = sensor_data.p(ordering, :);
time_s = (sensor.record_start_index:Nt) * dt_s;
design = [cos(omega * time_s(:)), sin(omega * time_s(:)), ones(numel(time_s), 1)];
coeff = design \ observer_pressure_Pa.';
observer_p_hat_Pa = (coeff(1, :) - 1i * coeff(2, :)).';
assert(all(isfinite(observer_pressure_Pa), 'all') ...
    && all(isfinite(observer_p_hat_Pa)), 'Non-finite propagated pressure.');
comp_info = getComputerInfo();
matlab_version = comp_info.matlab_version;
kwave_version = comp_info.kwave_version;
source_kappa_mode = 'additive: k-Wave cos(c*k*dt/2), no external correction';
save(output_mat_path, 'observer_p_hat_Pa', 'observer_pressure_Pa', 'time_s', ...
    'observer_x_m', 'observer_y_m', 'requested_observer_x_m', ...
    'requested_observer_y_m', 'x_grid_m', 'y_grid_m', 'pressure_source_hat', ...
    'ux_source_hat', 'uy_source_hat', 'dt_s', 'dx_m', 'Nt', 'points_per_cycle', ...
    'frequency_Hz', 'rho0_kg_m3', 'c0_m_s', 'source_kappa_mode', ...
    'diagnostic_mode', 'source_validation_passed', 'source_validation_bypassed', ...
    'kwave_backend_requested', 'kwave_backend_effective', 'data_cast', 'gpu_provenance_json', ...
    'elapsed_s', 'matlab_version', 'kwave_version', '-v7');
fprintf('Closed-curve propagation completed in %.3f s\n', elapsed_s);
end

function signal = harmonic(time, omega, ramp_time)
envelope = 0.5 * (1 - cos(pi * min(time / ramp_time, 1)));
signal = envelope .* exp(1i * omega * time);
end

function grid = deposit(x, y, values, x_axis, y_axis, dx)
% Conservative bilinear deposition, separately on each staggered lattice.
fx = (x - x_axis(1)) / dx + 1;
fy = (y - y_axis(1)) / dx + 1;
ix = floor(fx); iy = floor(fy);
wx = fx - ix; wy = fy - iy;
assert(all(ix >= 1 & ix < numel(x_axis) & iy >= 1 & iy < numel(y_axis)), ...
    'The handoff curve needs a one-cell margin inside the physical grid.');
indices = [sub2ind([numel(x_axis), numel(y_axis)], ix, iy); ...
    sub2ind([numel(x_axis), numel(y_axis)], ix+1, iy); ...
    sub2ind([numel(x_axis), numel(y_axis)], ix, iy+1); ...
    sub2ind([numel(x_axis), numel(y_axis)], ix+1, iy+1)];
weights = [(1-wx).*(1-wy); wx.*(1-wy); (1-wx).*wy; wx.*wy];
entries = repmat(values, 4, 1) .* weights;
grid = reshape(accumarray(indices, real(entries), [numel(x_axis)*numel(y_axis), 1]) ...
    + 1i * accumarray(indices, imag(entries), [numel(x_axis)*numel(y_axis), 1]), ...
    numel(x_axis), numel(y_axis));
assert(abs(sum(grid, 'all') - sum(values)) <= 1e-12 * max(sum(abs(values)), eps), ...
    'Curve deposition did not preserve the integrated source.');
end
