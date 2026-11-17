/*
 * Minimal two-dimensional CAC--solute reference implementation.
 *
 * Scope: periodic, no-flow subsystem only.  This program intentionally
 * contains no Navier--Stokes variables, pressure projection, multigrid code,
 * historical I/O layout, or platform-specific absolute paths.
 *
 * Spatial discretization: cell-centered second-order finite differences.
 * Time discretization: backward Euler start followed by BDF2.
 * Solute stabilization: Scheme B, Sc / D(phi*).
 * E-SAV: exponential update with the energy-optimized correction.
 *
 * The periodic phase Helmholtz problem is solved by a matrix-free conjugate
 * gradient method.  Under Scheme B the no-flow solute update is pointwise.
 */

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define MKDIR(path) mkdir(path, 0777)
#endif

#define PATH_CAP 512

typedef enum {
    INIT_TWO_DROPLETS = 0,
    INIT_SMOOTH = 1
} InitialCondition;

typedef struct {
    int nx;
    int ny;
    int steps;
    int cg_max_iter;
    double x0;
    double x1;
    double y0;
    double y1;
    double dt;
    double epsilon;
    double Sphi;
    double Sc;
    double alpha1;
    double alpha2;
    double beta1;
    double beta2;
    double D1;
    double D2;
    double Pe_phi;
    double Pe_c;
    double C_esav;
    double cg_tol;
    InitialCondition initial_condition;
    int write_fields;
    int quiet;
    char case_name[64];
    char output_dir[PATH_CAP];
} Config;

typedef struct {
    int status;
    int completed_steps;
    int max_cg_iterations;
    double final_time;
    double mass_phi_initial;
    double mass_phi_final;
    double mass_c_initial;
    double mass_c_final;
    double max_mass_phi_error;
    double max_mass_c_error;
    double min_c_star;
    double min_c;
    double min_D_star;
    double max_phase_relative_residual;
    double max_solute_relative_residual;
    double max_cg_relative_residual;
    double max_G1_zero_mean_error;
    double max_DG2_zero_mean_error;
    double energy_original_initial;
    double energy_original_final;
    double energy_corrected_initial;
    double energy_corrected_final;
    double max_corrected_energy_increase;
    double max_abs_Q_minus_one;
} RunSummary;

typedef struct {
    int iterations;
    int converged;
    double relative_residual;
} SolverInfo;

static void die(const char *message)
{
    fprintf(stderr, "ERROR: %s\n", message);
    exit(EXIT_FAILURE);
}

static double *allocate_field(size_t n)
{
    double *p = (double *)calloc(n, sizeof(double));
    if (!p) {
        die("memory allocation failed");
    }
    return p;
}

static void copy_field(double *dst, const double *src, size_t n)
{
    memcpy(dst, src, n * sizeof(double));
}

static double dot_product(const double *a, const double *b, size_t n)
{
    double sum = 0.0;
    size_t k;
    for (k = 0; k < n; ++k) {
        sum += a[k] * b[k];
    }
    return sum;
}

static double min_field(const double *a, size_t n)
{
    double value = DBL_MAX;
    size_t k;
    for (k = 0; k < n; ++k) {
        if (a[k] < value) {
            value = a[k];
        }
    }
    return value;
}

static int create_directories(const char *path)
{
    char buffer[PATH_CAP];
    size_t i;
    size_t length;

    if (!path || !path[0]) {
        return 0;
    }
    if (strlen(path) >= sizeof(buffer)) {
        return -1;
    }
    strcpy(buffer, path);
    length = strlen(buffer);

    for (i = 1; i < length; ++i) {
        if (buffer[i] == '/' || buffer[i] == '\\') {
            char saved = buffer[i];
            if (i == 2 && buffer[1] == ':') {
                continue;
            }
            buffer[i] = '\0';
            if (buffer[0] && MKDIR(buffer) != 0 && errno != EEXIST) {
                return -1;
            }
            buffer[i] = saved;
        }
    }
    if (MKDIR(buffer) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static void join_path(char *dst, size_t capacity, const char *dir, const char *name)
{
    const size_t length = strlen(dir);
    const char sep = (length > 0 && (dir[length - 1] == '/' || dir[length - 1] == '\\')) ? '\0' : '/';
    if (sep == '\0') {
        snprintf(dst, capacity, "%s%s", dir, name);
    } else {
        snprintf(dst, capacity, "%s%c%s", dir, sep, name);
    }
}

static void laplacian_periodic(const double *u, double *lap,
                               int nx, int ny, double dx, double dy)
{
    const double idx2 = 1.0 / (dx * dx);
    const double idy2 = 1.0 / (dy * dy);
    int i, j;
    for (j = 0; j < ny; ++j) {
        const int jm = (j + ny - 1) % ny;
        const int jp = (j + 1) % ny;
        for (i = 0; i < nx; ++i) {
            const int im = (i + nx - 1) % nx;
            const int ip = (i + 1) % nx;
            const size_t k = (size_t)j * (size_t)nx + (size_t)i;
            lap[k] = (u[(size_t)j * nx + ip] - 2.0 * u[k] + u[(size_t)j * nx + im]) * idx2
                   + (u[(size_t)jp * nx + i] - 2.0 * u[k] + u[(size_t)jm * nx + i]) * idy2;
        }
    }
}

static void apply_helmholtz(const double *u, double *Au, double *work_lap,
                            int nx, int ny, double dx, double dy,
                            double diagonal, double laplace_coefficient)
{
    size_t k;
    const size_t n = (size_t)nx * (size_t)ny;
    laplacian_periodic(u, work_lap, nx, ny, dx, dy);
    for (k = 0; k < n; ++k) {
        Au[k] = diagonal * u[k] - laplace_coefficient * work_lap[k];
    }
}

static SolverInfo solve_helmholtz_cg(double *x, const double *b,
                                     int nx, int ny, double dx, double dy,
                                     double diagonal, double laplace_coefficient,
                                     double tolerance, int max_iterations)
{
    const size_t n = (size_t)nx * (size_t)ny;
    double *r = allocate_field(n);
    double *p = allocate_field(n);
    double *Ap = allocate_field(n);
    double *work_lap = allocate_field(n);
    const double norm_b = sqrt(dot_product(b, b, n));
    const double denominator = (norm_b > DBL_MIN) ? norm_b : 1.0;
    double rr;
    SolverInfo info;
    int iter;
    size_t k;

    info.iterations = 0;
    info.converged = 0;
    info.relative_residual = DBL_MAX;

    apply_helmholtz(x, Ap, work_lap, nx, ny, dx, dy, diagonal, laplace_coefficient);
    for (k = 0; k < n; ++k) {
        r[k] = b[k] - Ap[k];
        p[k] = r[k];
    }
    rr = dot_product(r, r, n);
    info.relative_residual = sqrt(rr) / denominator;
    if (info.relative_residual <= tolerance) {
        info.converged = 1;
    }

    for (iter = 1; iter <= max_iterations && !info.converged; ++iter) {
        double alpha;
        double rr_new;
        double pAp;

        apply_helmholtz(p, Ap, work_lap, nx, ny, dx, dy, diagonal, laplace_coefficient);
        pAp = dot_product(p, Ap, n);
        if (!(pAp > 0.0) || !isfinite(pAp)) {
            break;
        }
        alpha = rr / pAp;
        for (k = 0; k < n; ++k) {
            x[k] += alpha * p[k];
            r[k] -= alpha * Ap[k];
        }
        rr_new = dot_product(r, r, n);
        info.iterations = iter;
        info.relative_residual = sqrt(rr_new) / denominator;
        if (info.relative_residual <= tolerance) {
            info.converged = 1;
            rr = rr_new;
            break;
        }
        if (!isfinite(rr_new) || rr <= DBL_MIN) {
            break;
        }
        {
            const double beta = rr_new / rr;
            for (k = 0; k < n; ++k) {
                p[k] = r[k] + beta * p[k];
            }
        }
        rr = rr_new;
    }

    /* Recompute the true algebraic residual instead of reporting only the
       recursively updated CG residual. */
    apply_helmholtz(x, Ap, work_lap, nx, ny, dx, dy, diagonal, laplace_coefficient);
    for (k = 0; k < n; ++k) {
        r[k] = b[k] - Ap[k];
    }
    info.relative_residual = sqrt(dot_product(r, r, n)) / denominator;
    info.converged = (info.relative_residual <= tolerance);

    free(r);
    free(p);
    free(Ap);
    free(work_lap);
    return info;
}

static double diffusion_coefficient(double phi, const Config *cfg)
{
    return 0.5 * (1.0 + phi) * cfg->D1 + 0.5 * (1.0 - phi) * cfg->D2;
}

static double bulk_energy_density(double phi, double c, const Config *cfg)
{
    const double logc = log(c);
    const double solute = 0.5 * (1.0 + phi) * cfg->alpha1 * c * (logc - 1.0 - cfg->beta1)
                        + 0.5 * (1.0 - phi) * cfg->alpha2 * c * (logc - 1.0 - cfg->beta2);
    const double phase = 0.25 * (phi * phi - 1.0) * (phi * phi - 1.0)
                       / (cfg->epsilon * cfg->epsilon);
    return solute + phase;
}

static double original_energy(const double *phi, const double *c, const Config *cfg)
{
    const double dx = (cfg->x1 - cfg->x0) / (double)cfg->nx;
    const double dy = (cfg->y1 - cfg->y0) / (double)cfg->ny;
    const double cell_area = dx * dy;
    double sum = 0.0;
    int i, j;
    for (j = 0; j < cfg->ny; ++j) {
        const int jp = (j + 1) % cfg->ny;
        for (i = 0; i < cfg->nx; ++i) {
            const int ip = (i + 1) % cfg->nx;
            const size_t k = (size_t)j * cfg->nx + i;
            const double gx = (phi[(size_t)j * cfg->nx + ip] - phi[k]) / dx;
            const double gy = (phi[(size_t)jp * cfg->nx + i] - phi[k]) / dy;
            if (!(c[k] > 0.0)) {
                return NAN;
            }
            sum += bulk_energy_density(phi[k], c[k], cfg) + 0.5 * (gx * gx + gy * gy);
        }
    }
    return cell_area * sum;
}

static double total_mass(const double *u, size_t n, double cell_area)
{
    double sum = 0.0;
    size_t k;
    for (k = 0; k < n; ++k) {
        sum += u[k];
    }
    return cell_area * sum;
}

static void compute_G_terms(const double *phi, const double *c,
                            double *D, double *G1, double *G2,
                            const Config *cfg,
                            double *G1_zero_mean_error,
                            double *DG2_zero_mean_error)
{
    const size_t n = (size_t)cfg->nx * (size_t)cfg->ny;
    const double dx = (cfg->x1 - cfg->x0) / (double)cfg->nx;
    const double dy = (cfg->y1 - cfg->y0) / (double)cfg->ny;
    const double cell_area = dx * dy;
    double sum_g1 = 0.0;
    double sum_Dg2 = 0.0;
    double sum_D = 0.0;
    double lambda1;
    double lambda2;
    size_t k;

    for (k = 0; k < n; ++k) {
        const double phi_k = phi[k];
        const double c_k = c[k];
        double g1;
        double g2;
        double logc;
        if (!(c_k > 0.0)) {
            die("nonpositive extrapolated concentration encountered before logarithm evaluation");
        }
        logc = log(c_k);
        D[k] = diffusion_coefficient(phi_k, cfg);
        if (!(D[k] > 0.0)) {
            die("nonpositive D(phi*) encountered");
        }
        g1 = (phi_k * phi_k * phi_k - phi_k) / (cfg->epsilon * cfg->epsilon)
           + 0.5 * cfg->alpha1 * c_k * (logc - 1.0 - cfg->beta1)
           - 0.5 * cfg->alpha2 * c_k * (logc - 1.0 - cfg->beta2);
        g2 = 0.5 * (1.0 + phi_k) * cfg->alpha1 * (logc - cfg->beta1)
           + 0.5 * (1.0 - phi_k) * cfg->alpha2 * (logc - cfg->beta2);
        G1[k] = g1;
        G2[k] = g2;
        sum_g1 += g1;
        sum_Dg2 += D[k] * g2;
        sum_D += D[k];
    }

    lambda1 = sum_g1 / (double)n;
    lambda2 = sum_Dg2 / sum_D;
    sum_g1 = 0.0;
    sum_Dg2 = 0.0;
    for (k = 0; k < n; ++k) {
        G1[k] -= lambda1;
        G2[k] -= lambda2;
        sum_g1 += G1[k];
        sum_Dg2 += D[k] * G2[k];
    }
    *G1_zero_mean_error = fabs(cell_area * sum_g1);
    *DG2_zero_mean_error = fabs(cell_area * sum_Dg2);
}

static void initialize_fields(double *phi, double *c, const Config *cfg)
{
    const double dx = (cfg->x1 - cfg->x0) / (double)cfg->nx;
    const double dy = (cfg->y1 - cfg->y0) / (double)cfg->ny;
    int i, j;
    for (j = 0; j < cfg->ny; ++j) {
        const double y = cfg->y0 + ((double)j + 0.5) * dy;
        for (i = 0; i < cfg->nx; ++i) {
            const double x = cfg->x0 + ((double)i + 0.5) * dx;
            const size_t k = (size_t)j * cfg->nx + i;
            if (cfg->initial_condition == INIT_TWO_DROPLETS) {
                const double r1 = hypot(x - 0.7, y - 1.0);
                const double r2 = hypot(x - 1.3, y - 1.0);
                phi[k] = tanh((0.29 - r1) / (sqrt(2.0) * cfg->epsilon))
                       + tanh((0.10 - r2) / (sqrt(2.0) * cfg->epsilon)) + 1.0;
                c[k] = 0.2;
            } else {
                const double pi = 4.0 * atan(1.0);
                phi[k] = 0.2 * sin(pi * x) * sin(pi * y);
                c[k] = 0.2 + 0.1 * cos(pi * x) * cos(pi * y);
            }
        }
    }
}

static void initialize_chemical_potentials(const double *phi, const double *c,
                                           double *mu_phi, double *mu_c,
                                           double *D, double *G1, double *G2,
                                           double *work_lap, const Config *cfg)
{
    const size_t n = (size_t)cfg->nx * (size_t)cfg->ny;
    const double dx = (cfg->x1 - cfg->x0) / (double)cfg->nx;
    const double dy = (cfg->y1 - cfg->y0) / (double)cfg->ny;
    double z1, z2;
    size_t k;
    compute_G_terms(phi, c, D, G1, G2, cfg, &z1, &z2);
    laplacian_periodic(phi, work_lap, cfg->nx, cfg->ny, dx, dy);
    for (k = 0; k < n; ++k) {
        mu_phi[k] = G1[k] - work_lap[k];
        mu_c[k] = G2[k];
    }
}

static double dissipation_at_star(const double *mu_phi_star,
                                  const double *mu_c_star,
                                  const double *D_star,
                                  size_t n, double cell_area,
                                  const Config *cfg)
{
    double sum = 0.0;
    size_t k;
    for (k = 0; k < n; ++k) {
        sum += mu_phi_star[k] * mu_phi_star[k] / cfg->Pe_phi
             + D_star[k] * mu_c_star[k] * mu_c_star[k] / cfg->Pe_c;
    }
    return cell_area * sum;
}

static double equation_relative_residual(const double *time_term,
                                         const double *chemical_term,
                                         double *residual,
                                         size_t n)
{
    double norm_r2 = 0.0;
    double scale2 = 0.0;
    size_t k;
    for (k = 0; k < n; ++k) {
        residual[k] = time_term[k] + chemical_term[k];
        norm_r2 += residual[k] * residual[k];
        scale2 += time_term[k] * time_term[k] + chemical_term[k] * chemical_term[k];
    }
    return sqrt(norm_r2) / ((sqrt(scale2) > DBL_MIN) ? sqrt(scale2) : 1.0);
}

static void write_final_fields(const char *path,
                               const double *phi, const double *c,
                               const Config *cfg)
{
    FILE *fp = fopen(path, "w");
    const double dx = (cfg->x1 - cfg->x0) / (double)cfg->nx;
    const double dy = (cfg->y1 - cfg->y0) / (double)cfg->ny;
    int i, j;
    if (!fp) {
        die("cannot open final field output");
    }
    fprintf(fp, "x,y,phi,c\n");
    for (j = 0; j < cfg->ny; ++j) {
        const double y = cfg->y0 + ((double)j + 0.5) * dy;
        for (i = 0; i < cfg->nx; ++i) {
            const double x = cfg->x0 + ((double)i + 0.5) * dx;
            const size_t k = (size_t)j * cfg->nx + i;
            fprintf(fp, "%.17g,%.17g,%.17g,%.17g\n", x, y, phi[k], c[k]);
        }
    }
    fclose(fp);
}

static void write_summary(const char *path, const Config *cfg, const RunSummary *s)
{
    FILE *fp = fopen(path, "w");
    if (!fp) {
        die("cannot open summary output");
    }
    fprintf(fp, "case=%s\n", cfg->case_name);
    fprintf(fp, "grid=%dx%d\n", cfg->nx, cfg->ny);
    fprintf(fp, "dt=%.17g\n", cfg->dt);
    fprintf(fp, "requested_steps=%d\n", cfg->steps);
    fprintf(fp, "completed_steps=%d\n", s->completed_steps);
    fprintf(fp, "D1=%.17g\nD2=%.17g\n", cfg->D1, cfg->D2);
    fprintf(fp, "status=%s\n", s->status == 0 ? "PASS" : "FAIL");
    fprintf(fp, "max_abs_mass_phi_error=%.17e\n", s->max_mass_phi_error);
    fprintf(fp, "max_abs_mass_c_error=%.17e\n", s->max_mass_c_error);
    fprintf(fp, "min_c_star=%.17e\n", s->min_c_star);
    fprintf(fp, "min_c=%.17e\n", s->min_c);
    fprintf(fp, "min_D_star=%.17e\n", s->min_D_star);
    fprintf(fp, "max_phase_relative_residual=%.17e\n", s->max_phase_relative_residual);
    fprintf(fp, "max_solute_relative_residual=%.17e\n", s->max_solute_relative_residual);
    fprintf(fp, "max_cg_relative_residual=%.17e\n", s->max_cg_relative_residual);
    fprintf(fp, "max_G1_zero_mean_error=%.17e\n", s->max_G1_zero_mean_error);
    fprintf(fp, "max_DG2_zero_mean_error=%.17e\n", s->max_DG2_zero_mean_error);
    fprintf(fp, "original_energy_initial=%.17e\n", s->energy_original_initial);
    fprintf(fp, "original_energy_final=%.17e\n", s->energy_original_final);
    fprintf(fp, "corrected_energy_initial=%.17e\n", s->energy_corrected_initial);
    fprintf(fp, "corrected_energy_final=%.17e\n", s->energy_corrected_final);
    fprintf(fp, "max_corrected_energy_increase=%.17e\n", s->max_corrected_energy_increase);
    fprintf(fp, "max_abs_Q_minus_one=%.17e\n", s->max_abs_Q_minus_one);
    fprintf(fp, "max_cg_iterations=%d\n", s->max_cg_iterations);
    fclose(fp);
}

static int run_simulation(const Config *cfg, RunSummary *summary,
                          double **final_phi_out, double **final_c_out)
{
    const size_t n = (size_t)cfg->nx * (size_t)cfg->ny;
    const double dx = (cfg->x1 - cfg->x0) / (double)cfg->nx;
    const double dy = (cfg->y1 - cfg->y0) / (double)cfg->ny;
    const double cell_area = dx * dy;
    double *phi_nm1 = allocate_field(n);
    double *phi_n = allocate_field(n);
    double *phi_new = allocate_field(n);
    double *c_nm1 = allocate_field(n);
    double *c_n = allocate_field(n);
    double *c_new = allocate_field(n);
    double *mu_phi_nm1 = allocate_field(n);
    double *mu_phi_n = allocate_field(n);
    double *mu_phi_new = allocate_field(n);
    double *mu_c_nm1 = allocate_field(n);
    double *mu_c_n = allocate_field(n);
    double *mu_c_new = allocate_field(n);
    double *phi_star = allocate_field(n);
    double *c_star = allocate_field(n);
    double *mu_phi_star = allocate_field(n);
    double *mu_c_star = allocate_field(n);
    double *D_star = allocate_field(n);
    double *G1 = allocate_field(n);
    double *G2 = allocate_field(n);
    double *rhs = allocate_field(n);
    double *lap = allocate_field(n);
    double *time_term = allocate_field(n);
    double *chemical_term = allocate_field(n);
    double *residual = allocate_field(n);
    FILE *diagnostics = NULL;
    char diagnostics_path[PATH_CAP];
    char fields_path[PATH_CAP];
    char summary_path[PATH_CAP];
    double V;
    double original_energy_n;
    double corrected_energy_n;
    int step;
    size_t k;

    memset(summary, 0, sizeof(*summary));
    summary->status = 1;
    summary->min_c_star = DBL_MAX;
    summary->min_c = DBL_MAX;
    summary->min_D_star = DBL_MAX;

    initialize_fields(phi_n, c_n, cfg);
    copy_field(phi_nm1, phi_n, n);
    copy_field(c_nm1, c_n, n);
    initialize_chemical_potentials(phi_n, c_n, mu_phi_n, mu_c_n,
                                   D_star, G1, G2, lap, cfg);
    copy_field(mu_phi_nm1, mu_phi_n, n);
    copy_field(mu_c_nm1, mu_c_n, n);

    original_energy_n = original_energy(phi_n, c_n, cfg);
    if (!isfinite(original_energy_n)) {
        die("invalid initial energy");
    }
    V = exp(original_energy_n / cfg->C_esav);
    corrected_energy_n = cfg->C_esav * log(V);

    summary->mass_phi_initial = total_mass(phi_n, n, cell_area);
    summary->mass_c_initial = total_mass(c_n, n, cell_area);
    summary->energy_original_initial = original_energy_n;
    summary->energy_corrected_initial = corrected_energy_n;
    summary->min_c = min_field(c_n, n);

    if (cfg->output_dir[0]) {
        if (create_directories(cfg->output_dir) != 0) {
            die("cannot create output directory");
        }
        join_path(diagnostics_path, sizeof(diagnostics_path), cfg->output_dir, "diagnostics.csv");
        diagnostics = fopen(diagnostics_path, "w");
        if (!diagnostics) {
            die("cannot open diagnostics.csv");
        }
        fprintf(diagnostics,
                "step,time,mass_phi_error,mass_c_error,min_c_star,min_c,min_D_star,"
                "original_energy,corrected_energy,Q,phase_relative_residual,"
                "solute_relative_residual,cg_relative_residual,cg_iterations,"
                "G1_zero_mean_error,DG2_zero_mean_error\n");
        fprintf(diagnostics,
                "0,0,0,0,%.17e,%.17e,%.17e,%.17e,%.17e,1,0,0,0,0,0,0\n",
                summary->min_c, summary->min_c, min_field(D_star, n),
                original_energy_n, corrected_energy_n);
    }

    for (step = 1; step <= cfg->steps; ++step) {
        const int backward_euler = (step == 1);
        const double time_coefficient = backward_euler ? 1.0 / cfg->dt : 3.0 / (2.0 * cfg->dt);
        const double phase_stabilization = cfg->Sphi
                                           / (cfg->Pe_phi * cfg->epsilon * cfg->epsilon);
        const double phase_diagonal = time_coefficient + phase_stabilization;
        const double phase_laplace_coefficient = 1.0 / cfg->Pe_phi;
        const double solute_denominator = time_coefficient + cfg->Sc / cfg->Pe_c;
        double G1_zero_mean_error;
        double DG2_zero_mean_error;
        double dissipation_star;
        double energy_star;
        double Vhat;
        double xi;
        double Q;
        double original_energy_new;
        double Vbar;
        double Vnew;
        double corrected_energy_new;
        double phase_relative_residual;
        double solute_relative_residual;
        double mass_phi;
        double mass_c;
        double min_c_star;
        double min_c_new;
        double min_D;
        SolverInfo solver;

        for (k = 0; k < n; ++k) {
            if (backward_euler) {
                phi_star[k] = phi_n[k];
                c_star[k] = c_n[k];
                mu_phi_star[k] = mu_phi_n[k];
                mu_c_star[k] = mu_c_n[k];
            } else {
                phi_star[k] = 2.0 * phi_n[k] - phi_nm1[k];
                c_star[k] = 2.0 * c_n[k] - c_nm1[k];
                mu_phi_star[k] = 2.0 * mu_phi_n[k] - mu_phi_nm1[k];
                mu_c_star[k] = 2.0 * mu_c_n[k] - mu_c_nm1[k];
            }
        }

        min_c_star = min_field(c_star, n);
        if (!(min_c_star > 0.0)) {
            fprintf(stderr, "step %d: min(c*) = %.17e is not positive\n", step, min_c_star);
            break;
        }
        compute_G_terms(phi_star, c_star, D_star, G1, G2, cfg,
                        &G1_zero_mean_error, &DG2_zero_mean_error);
        min_D = min_field(D_star, n);
        energy_star = original_energy(phi_star, c_star, cfg);
        dissipation_star = dissipation_at_star(mu_phi_star, mu_c_star, D_star,
                                               n, cell_area, cfg);
        Vhat = V / (1.0 + cfg->dt * dissipation_star / cfg->C_esav);
        xi = Vhat * exp(-energy_star / cfg->C_esav);
        Q = backward_euler ? xi : xi * (2.0 - xi);

        for (k = 0; k < n; ++k) {
            const double history_phi = backward_euler
                                     ? phi_n[k] / cfg->dt
                                     : (4.0 * phi_n[k] - phi_nm1[k]) / (2.0 * cfg->dt);
            rhs[k] = history_phi - Q * G1[k] / cfg->Pe_phi
                   + phase_stabilization * phi_star[k];
            phi_new[k] = phi_n[k];
        }
        solver = solve_helmholtz_cg(phi_new, rhs, cfg->nx, cfg->ny, dx, dy,
                                    phase_diagonal, phase_laplace_coefficient,
                                    cfg->cg_tol, cfg->cg_max_iter);
        if (!solver.converged) {
            fprintf(stderr, "step %d: phase CG failed, relative residual %.17e\n",
                    step, solver.relative_residual);
            break;
        }

        for (k = 0; k < n; ++k) {
            const double history_c = backward_euler
                                   ? c_n[k] / cfg->dt
                                   : (4.0 * c_n[k] - c_nm1[k]) / (2.0 * cfg->dt);
            c_new[k] = (history_c - Q * D_star[k] * G2[k] / cfg->Pe_c
                        + (cfg->Sc / cfg->Pe_c) * c_star[k]) / solute_denominator;
        }
        min_c_new = min_field(c_new, n);
        if (!(min_c_new > 0.0)) {
            fprintf(stderr, "step %d: min(c) = %.17e is not positive\n", step, min_c_new);
            break;
        }

        laplacian_periodic(phi_new, lap, cfg->nx, cfg->ny, dx, dy);
        for (k = 0; k < n; ++k) {
            mu_phi_new[k] = Q * G1[k] - lap[k]
                          + cfg->Sphi / (cfg->epsilon * cfg->epsilon)
                          * (phi_new[k] - phi_star[k]);
            mu_c_new[k] = Q * G2[k] + cfg->Sc / D_star[k] * (c_new[k] - c_star[k]);

            time_term[k] = backward_euler
                         ? (phi_new[k] - phi_n[k]) / cfg->dt
                         : (3.0 * phi_new[k] - 4.0 * phi_n[k] + phi_nm1[k])
                           / (2.0 * cfg->dt);
            chemical_term[k] = mu_phi_new[k] / cfg->Pe_phi;
        }
        phase_relative_residual = equation_relative_residual(time_term, chemical_term,
                                                             residual, n);

        for (k = 0; k < n; ++k) {
            time_term[k] = backward_euler
                         ? (c_new[k] - c_n[k]) / cfg->dt
                         : (3.0 * c_new[k] - 4.0 * c_n[k] + c_nm1[k])
                           / (2.0 * cfg->dt);
            chemical_term[k] = D_star[k] * mu_c_new[k] / cfg->Pe_c;
        }
        solute_relative_residual = equation_relative_residual(time_term, chemical_term,
                                                              residual, n);

        original_energy_new = original_energy(phi_new, c_new, cfg);
        Vbar = exp(original_energy_new / cfg->C_esav);
        Vnew = (V < Vbar) ? V : Vbar;
        corrected_energy_new = cfg->C_esav * log(Vnew);

        mass_phi = total_mass(phi_new, n, cell_area);
        mass_c = total_mass(c_new, n, cell_area);
        if (fabs(mass_phi - summary->mass_phi_initial) > summary->max_mass_phi_error) {
            summary->max_mass_phi_error = fabs(mass_phi - summary->mass_phi_initial);
        }
        if (fabs(mass_c - summary->mass_c_initial) > summary->max_mass_c_error) {
            summary->max_mass_c_error = fabs(mass_c - summary->mass_c_initial);
        }
        if (min_c_star < summary->min_c_star) summary->min_c_star = min_c_star;
        if (min_c_new < summary->min_c) summary->min_c = min_c_new;
        if (min_D < summary->min_D_star) summary->min_D_star = min_D;
        if (phase_relative_residual > summary->max_phase_relative_residual)
            summary->max_phase_relative_residual = phase_relative_residual;
        if (solute_relative_residual > summary->max_solute_relative_residual)
            summary->max_solute_relative_residual = solute_relative_residual;
        if (solver.relative_residual > summary->max_cg_relative_residual)
            summary->max_cg_relative_residual = solver.relative_residual;
        if (solver.iterations > summary->max_cg_iterations)
            summary->max_cg_iterations = solver.iterations;
        if (G1_zero_mean_error > summary->max_G1_zero_mean_error)
            summary->max_G1_zero_mean_error = G1_zero_mean_error;
        if (DG2_zero_mean_error > summary->max_DG2_zero_mean_error)
            summary->max_DG2_zero_mean_error = DG2_zero_mean_error;
        if (corrected_energy_new - corrected_energy_n > summary->max_corrected_energy_increase)
            summary->max_corrected_energy_increase = corrected_energy_new - corrected_energy_n;
        if (fabs(Q - 1.0) > summary->max_abs_Q_minus_one)
            summary->max_abs_Q_minus_one = fabs(Q - 1.0);

        if (diagnostics) {
            fprintf(diagnostics,
                    "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,"
                    "%.17e,%.17e,%.17e,%.17e,%d,%.17e,%.17e\n",
                    step, step * cfg->dt,
                    mass_phi - summary->mass_phi_initial,
                    mass_c - summary->mass_c_initial,
                    min_c_star, min_c_new, min_D,
                    original_energy_new, corrected_energy_new, Q,
                    phase_relative_residual, solute_relative_residual,
                    solver.relative_residual, solver.iterations,
                    G1_zero_mean_error, DG2_zero_mean_error);
        }

        copy_field(phi_nm1, phi_n, n);
        copy_field(phi_n, phi_new, n);
        copy_field(c_nm1, c_n, n);
        copy_field(c_n, c_new, n);
        copy_field(mu_phi_nm1, mu_phi_n, n);
        copy_field(mu_phi_n, mu_phi_new, n);
        copy_field(mu_c_nm1, mu_c_n, n);
        copy_field(mu_c_n, mu_c_new, n);
        V = Vnew;
        original_energy_n = original_energy_new;
        corrected_energy_n = corrected_energy_new;
        summary->completed_steps = step;
    }

    if (diagnostics) {
        fclose(diagnostics);
    }

    summary->final_time = summary->completed_steps * cfg->dt;
    summary->mass_phi_final = total_mass(phi_n, n, cell_area);
    summary->mass_c_final = total_mass(c_n, n, cell_area);
    summary->energy_original_final = original_energy_n;
    summary->energy_corrected_final = corrected_energy_n;
    summary->status = (summary->completed_steps == cfg->steps) ? 0 : 1;

    if (cfg->output_dir[0]) {
        join_path(summary_path, sizeof(summary_path), cfg->output_dir, "summary.txt");
        write_summary(summary_path, cfg, summary);
        if (cfg->write_fields) {
            join_path(fields_path, sizeof(fields_path), cfg->output_dir, "final_fields.csv");
            write_final_fields(fields_path, phi_n, c_n, cfg);
        }
    }

    if (final_phi_out) {
        *final_phi_out = allocate_field(n);
        copy_field(*final_phi_out, phi_n, n);
    }
    if (final_c_out) {
        *final_c_out = allocate_field(n);
        copy_field(*final_c_out, c_n, n);
    }

    free(phi_nm1); free(phi_n); free(phi_new);
    free(c_nm1); free(c_n); free(c_new);
    free(mu_phi_nm1); free(mu_phi_n); free(mu_phi_new);
    free(mu_c_nm1); free(mu_c_n); free(mu_c_new);
    free(phi_star); free(c_star); free(mu_phi_star); free(mu_c_star);
    free(D_star); free(G1); free(G2); free(rhs); free(lap);
    free(time_term); free(chemical_term); free(residual);
    return summary->status;
}

static void print_run_summary(const Config *cfg, const RunSummary *s)
{
    printf("case: %s\n", cfg->case_name);
    printf("status: %s (%d/%d steps, t=%.8g)\n",
           s->status == 0 ? "PASS" : "FAIL",
           s->completed_steps, cfg->steps, s->final_time);
    printf("max |M_phi-M_phi0|: %.6e\n", s->max_mass_phi_error);
    printf("max |M_c-M_c0|:     %.6e\n", s->max_mass_c_error);
    printf("min(c*), min(c), min(D*): %.6e  %.6e  %.6e\n",
           s->min_c_star, s->min_c, s->min_D_star);
    printf("max relative residual (phi,c,CG): %.6e  %.6e  %.6e\n",
           s->max_phase_relative_residual,
           s->max_solute_relative_residual,
           s->max_cg_relative_residual);
    printf("max zero-mean errors (G1,DG2): %.6e  %.6e\n",
           s->max_G1_zero_mean_error, s->max_DG2_zero_mean_error);
    printf("original energy:  %.12e -> %.12e\n",
           s->energy_original_initial, s->energy_original_final);
    printf("corrected energy: %.12e -> %.12e (max increase %.3e)\n",
           s->energy_corrected_initial, s->energy_corrected_final,
           s->max_corrected_energy_increase);
    printf("max |Q-1|: %.6e; max CG iterations: %d\n",
           s->max_abs_Q_minus_one, s->max_cg_iterations);
}

static Config default_config(void)
{
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.nx = 256;
    cfg.ny = 256;
    cfg.steps = 160;
    cfg.cg_max_iter = 2000;
    cfg.x0 = 0.0;
    cfg.x1 = 2.0;
    cfg.y0 = 0.0;
    cfg.y1 = 2.0;
    cfg.dt = 0.01;
    cfg.epsilon = 0.018;
    cfg.Sphi = 6.0;
    cfg.Sc = 6.0;
    cfg.alpha1 = 1.0;
    cfg.alpha2 = 1.0;
    cfg.beta1 = log(0.4);
    cfg.beta2 = log(0.8);
    cfg.D1 = 1.0;
    cfg.D2 = 1.0;
    cfg.Pe_phi = 1.0 / cfg.epsilon;
    cfg.Pe_c = 0.02;
    cfg.C_esav = 1.0e4;
    cfg.cg_tol = 1.0e-12;
    cfg.initial_condition = INIT_TWO_DROPLETS;
    cfg.write_fields = 1;
    strcpy(cfg.case_name, "equal_D");
    strcpy(cfg.output_dir, "results_equal_D");
    return cfg;
}

static double l2_error(const double *a, const double *b, size_t n, double cell_area)
{
    double sum = 0.0;
    size_t k;
    for (k = 0; k < n; ++k) {
        const double d = a[k] - b[k];
        sum += d * d;
    }
    return sqrt(cell_area * sum);
}

static int run_convergence_test(const char *output_dir)
{
    const double dts[] = {1.0e-3, 5.0e-4, 2.5e-4};
    const int count = 3;
    const double final_time = 2.0e-2;
    const double reference_dt = 3.125e-5;
    Config cfg = default_config();
    RunSummary summary;
    double *phi_ref = NULL;
    double *c_ref = NULL;
    double phi_errors[3];
    double c_errors[3];
    char path[PATH_CAP];
    FILE *fp;
    int m;

    cfg.nx = 64;
    cfg.ny = 64;
    cfg.x0 = -1.0;
    cfg.x1 = 1.0;
    cfg.y0 = -1.0;
    cfg.y1 = 1.0;
    cfg.initial_condition = INIT_SMOOTH;
    cfg.write_fields = 0;
    cfg.quiet = 1;
    cfg.output_dir[0] = '\0';
    strcpy(cfg.case_name, "time_reference");
    cfg.dt = reference_dt;
    cfg.steps = (int)llround(final_time / reference_dt);
    if (run_simulation(&cfg, &summary, &phi_ref, &c_ref) != 0) {
        fprintf(stderr, "time-convergence reference run failed\n");
        free(phi_ref); free(c_ref);
        return 1;
    }

    for (m = 0; m < count; ++m) {
        double *phi = NULL;
        double *c = NULL;
        const size_t n = (size_t)cfg.nx * (size_t)cfg.ny;
        const double cell_area = (cfg.x1 - cfg.x0) * (cfg.y1 - cfg.y0) / (double)n;
        cfg.dt = dts[m];
        cfg.steps = (int)llround(final_time / cfg.dt);
        snprintf(cfg.case_name, sizeof(cfg.case_name), "time_dt_%.4g", cfg.dt);
        if (run_simulation(&cfg, &summary, &phi, &c) != 0) {
            fprintf(stderr, "time-convergence run failed for dt=%.8g\n", cfg.dt);
            free(phi); free(c); free(phi_ref); free(c_ref);
            return 1;
        }
        phi_errors[m] = l2_error(phi, phi_ref, n, cell_area);
        c_errors[m] = l2_error(c, c_ref, n, cell_area);
        free(phi);
        free(c);
    }

    if (create_directories(output_dir) != 0) {
        die("cannot create convergence output directory");
    }
    join_path(path, sizeof(path), output_dir, "convergence.csv");
    fp = fopen(path, "w");
    if (!fp) {
        die("cannot open convergence.csv");
    }
    fprintf(fp, "dt,phi_L2_error,phi_rate,c_L2_error,c_rate\n");
    for (m = 0; m < count; ++m) {
        const double phi_rate = (m == 0) ? NAN : log(phi_errors[m - 1] / phi_errors[m]) / log(2.0);
        const double c_rate = (m == 0) ? NAN : log(c_errors[m - 1] / c_errors[m]) / log(2.0);
        fprintf(fp, "%.17e,%.17e,", dts[m], phi_errors[m]);
        if (m == 0) fprintf(fp, ","); else fprintf(fp, "%.10f,", phi_rate);
        fprintf(fp, "%.17e,", c_errors[m]);
        if (m == 0) fprintf(fp, "\n"); else fprintf(fp, "%.10f\n", c_rate);
    }
    fclose(fp);

    printf("time convergence (reference dt %.6g, T %.6g, grid 64x64)\n",
           reference_dt, final_time);
    for (m = 0; m < count; ++m) {
        if (m == 0) {
            printf("dt %.6g: phi %.6e, c %.6e\n", dts[m], phi_errors[m], c_errors[m]);
        } else {
            printf("dt %.6g: phi %.6e (rate %.4f), c %.6e (rate %.4f)\n",
                   dts[m], phi_errors[m],
                   log(phi_errors[m - 1] / phi_errors[m]) / log(2.0),
                   c_errors[m], log(c_errors[m - 1] / c_errors[m]) / log(2.0));
        }
    }

    free(phi_ref);
    free(c_ref);
    return 0;
}

static void print_usage(const char *program)
{
    printf("Usage:\n");
    printf("  %s [--case equal|variable] [--nx N] [--dt DT] [--steps N]\n", program);
    printf("     [--output DIR] [--no-fields] [--quiet]\n");
    printf("  %s --convergence [--output DIR]\n", program);
}

int main(int argc, char **argv)
{
    Config cfg = default_config();
    RunSummary summary;
    int convergence = 0;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--case") == 0 && i + 1 < argc) {
            const char *name = argv[++i];
            if (strcmp(name, "equal") == 0) {
                cfg.D1 = 1.0;
                cfg.D2 = 1.0;
                strcpy(cfg.case_name, "equal_D");
                strcpy(cfg.output_dir, "results_equal_D");
            } else if (strcmp(name, "variable") == 0) {
                cfg.D1 = 1.0;
                cfg.D2 = 0.5;
                strcpy(cfg.case_name, "variable_D");
                strcpy(cfg.output_dir, "results_variable_D");
            } else {
                die("unknown case; use equal or variable");
            }
        } else if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            cfg.nx = atoi(argv[++i]);
            cfg.ny = cfg.nx;
        } else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
            cfg.dt = atof(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            cfg.steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            strncpy(cfg.output_dir, argv[++i], sizeof(cfg.output_dir) - 1);
            cfg.output_dir[sizeof(cfg.output_dir) - 1] = '\0';
        } else if (strcmp(argv[i], "--no-fields") == 0) {
            cfg.write_fields = 0;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            cfg.quiet = 1;
        } else if (strcmp(argv[i], "--convergence") == 0) {
            convergence = 1;
            strcpy(cfg.output_dir, "results_convergence");
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.nx < 8 || cfg.ny < 8 || cfg.steps < 1 || !(cfg.dt > 0.0)) {
        die("invalid grid or time-step setting");
    }

    if (convergence) {
        return run_convergence_test(cfg.output_dir);
    }

    if (run_simulation(&cfg, &summary, NULL, NULL) != 0) {
        print_run_summary(&cfg, &summary);
        return 2;
    }
    if (!cfg.quiet) {
        print_run_summary(&cfg, &summary);
    }
    return 0;
}
