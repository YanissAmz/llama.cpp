// M5 test: the tile-aware ESCHAM branch of ggml_compute_forward_mul_mat.
//
// Same fixture and same reference as test-escha-m4, but the weight is no
// longer a dense F16 tensor: it is a GGML_TYPE_ESCHAM_2/3 tensor holding the
// packed codes band-major, and the product goes through the CPU kernel that
// decodes one 16-output band at a time.
//
// The two Hadamards stay in plain C (escham_vec_had128_f32) so that a failure
// points at the matmul branch and nothing else. M4 already proved the
// transforms and the tile decoder separately.
//
// Usage: test-escha-m5 [fixture]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-escham.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_DEFAULT "fixture_m4.bin"
#define FIXTURE_MAGIC 0x45534D34u // 'ESM4'
#define CRIT 1e-3f

static const uint8_t *cur;
static const uint8_t *take(size_t n) { const uint8_t *p = cur; cur += n; return p; }
static uint32_t rd_u32(void) { uint32_t v; memcpy(&v, take(4), 4); return v; }

static float rel_rms(const float *a, const float *b, size_t n) {
    double se = 0.0, sb = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        se += d * d;
        sb += (double)b[i] * (double)b[i];
    }
    return (float)(sqrt(se / n) / (sqrt(sb / n) + 1e-12));
}

static void *read_all(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return NULL; }
    fclose(f); *n = (size_t)sz; return buf;
}

static int test_record(const char *name, int K, int IC, int OC, int NR,
                       const uint16_t *packed, const float *x, const float *rin,
                       const float *rout, const float *bias, const float *ref_y) {
    const int64_t itc = IC / 16, otc = OC / 16;
    const size_t tile_u16 = (size_t)16 * K;

    // ---- weight: checkpoint order (itc, otc, 16K) -> band-major (otc, itc, 16K)
    uint16_t *band = malloc((size_t)itc * otc * tile_u16 * sizeof(uint16_t));
    if (!band) { fprintf(stderr, "%s: OOM\n", name); return 1; }
    for (int64_t ot = 0; ot < otc; ot++) {
        for (int64_t it = 0; it < itc; it++) {
            memcpy(band + (ot * itc + it) * tile_u16,
                   packed + (it * otc + ot) * tile_u16,
                   tile_u16 * sizeof(uint16_t));
        }
    }

    size_t pool = (size_t)itc * otc * tile_u16 * 2 + (size_t)(IC + OC) * NR * 4 * 4
                + 64ull * 1024 * 1024;
    struct ggml_init_params ip = { .mem_size = pool, .mem_buffer = NULL, .no_alloc = false };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *tw = ggml_new_tensor_2d(
        ctx, (K == 3) ? GGML_TYPE_ESCHAM_3 : GGML_TYPE_ESCHAM_2, IC, OC);
    if (ggml_nbytes(tw) != (size_t)itc * otc * tile_u16 * sizeof(uint16_t)) {
        fprintf(stderr, "%s: ggml_nbytes %zu != packed %zu -- block traits are wrong\n",
                name, ggml_nbytes(tw), (size_t)itc * otc * tile_u16 * sizeof(uint16_t));
        free(band); ggml_free(ctx); return 1;
    }
    memcpy(tw->data, band, ggml_nbytes(tw));
    free(band);

    // ---- xp = T128(x * rin), column j of src1 is sample j
    struct ggml_tensor *txp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IC, NR);
    for (int64_t j = 0; j < NR; j++) {
        float *col = (float *)txp->data + j * IC;
        for (int64_t i = 0; i < IC; i++) {
            col[i] = x[i * NR + j] * rin[i];
        }
        escham_vec_had128_f32(col, IC);
    }

    struct ggml_tensor *z = ggml_mul_mat(ctx, tw, txp);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, z);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) { fprintf(stderr, "%s: no cpu backend\n", name); ggml_free(ctx); return 1; }
    const char *nth_env = getenv("ESCHAM_M5_NTH");
    ggml_backend_cpu_set_n_threads(cpu, nth_env ? atoi(nth_env) : 4);
    enum ggml_status st = ggml_backend_graph_compute(cpu, gf);
    ggml_backend_free(cpu);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: graph compute failed (%d)\n", name, (int)st);
        ggml_free(ctx); return 1;
    }

    // ---- y = T128(z) * rout + bias, laid out like the fixture: ref[o*NR + j]
    float *got = malloc((size_t)OC * NR * sizeof(float));
    if (!got) { fprintf(stderr, "%s: OOM\n", name); ggml_free(ctx); return 1; }
    for (int64_t j = 0; j < NR; j++) {
        float *col = (float *)z->data + j * OC;
        escham_vec_had128_f32(col, OC);
        for (int64_t o = 0; o < OC; o++) {
            got[o * NR + j] = col[o] * rout[o] + bias[o];
        }
    }

    const float rms = rel_rms(got, ref_y, (size_t)OC * NR);
    free(got);
    ggml_free(ctx);

    printf("%s: escham mul_mat K=%d IC=%d OC=%d NR=%d  rms rel %.3e  %s\n",
           name, K, IC, OC, NR, rms, (rms < CRIT) ? "OK" : "FAIL");
    return (rms < CRIT) ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *env = getenv("ESCHA_FIXTURE");
    const char *path = (argc > 1) ? argv[1] : (env && env[0]) ? env : FIXTURE_DEFAULT;
    size_t n = 0;
    uint8_t *buf = read_all(path, &n);
    if (!buf) return 1;
    cur = buf;

    if (rd_u32() != FIXTURE_MAGIC) { fprintf(stderr, "bad magic in %s\n", path); return 1; }
    (void)rd_u32(); // version
    const uint32_t nrec = rd_u32();
    int fails = 0;
    for (uint32_t r = 0; r < nrec; r++) {
        const uint32_t nl = rd_u32();
        char name[512];
        const uint32_t cl = (nl < sizeof(name) - 1) ? nl : (uint32_t)sizeof(name) - 1;
        memcpy(name, take(nl), nl);
        name[cl] = 0;

        int32_t hdr[5];
        memcpy(hdr, take(sizeof(hdr)), sizeof(hdr));
        const int K = hdr[0], IC = hdr[1], OC = hdr[2], NT = hdr[3], NR = hdr[4];

        take((size_t)NT * 16 * K * sizeof(uint16_t));   // codes_unit, M4 only
        take((size_t)NT * 256 * sizeof(uint16_t));      // exp_unit,   M4 only
        const uint16_t *packed = (const uint16_t *)take((size_t)IC * OC * K / 16 * sizeof(uint16_t));
        const float *x     = (const float *)take((size_t)IC * NR * sizeof(float));
        const float *rin   = (const float *)take((size_t)IC * sizeof(float));
        const float *rout  = (const float *)take((size_t)OC * sizeof(float));
        const float *bias  = (const float *)take((size_t)OC * sizeof(float));
        const float *ref_y = (const float *)take((size_t)OC * NR * sizeof(float));

        printf("-- %s: K=%d IC=%d OC=%d NR=%d\n", name, K, IC, OC, NR);
        fails += test_record(name, K, IC, OC, NR, packed, x, rin, rout, bias, ref_y);
    }
    free(buf);
    printf("\nverdict: %s (%d failing checks)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
