"""
Single source of truth for the ablation grid.

Each value names one module or one prior to remove. The loss-cluster
ablations are deliberately absent: `--lambda-prob/-proto/-rel 0` already
disable those (their help text says so), and they reuse the baseline
teacher, so they need no entry here.

An ablated model keeps every parameter the baseline has. Disabling a
prior leaves theta_beta / reliability_mlp / eta allocated but unused, and
dropping hyperedges leaves w_q/w_k/w_v/w_o untouched. So state_dict
shapes are identical across the whole grid, which means an ablated
checkpoint would load silently into a baseline model -- the tag in the
checkpoint filename is the only thing keeping them apart. That is why
load_teacher_checkpoint resolves the path from --teacher-ablate rather
than trusting whatever is on disk.
"""

# 'no-priors' is the middle rung of the context-graph ladder:
#   baseline         logit = content + log(w_T * w_S * w_R)
#   no-priors        logit = content                 <- plain windowed attention
#   no-context-graph no attention at all
# The three single-prior ablations cannot substitute for it, because the
# priors multiply and can compensate for one another.
ABLATE_CHOICES = (
    'none',
    # --- group 1: context graph ---
    'no-context-graph',   # skip the attention layers, keep the [N,D] -> [T,B,D] reshape
    'no-priors',          # all three at once: log_prior = 0, i.e. plain windowed attention
    'no-temporal',        # w_T = 1  (drops the LEARNED decay; the hard +-window stays)
    'no-speaker',         # w_S = 1
    'no-uncertainty',     # w_R = 1
    # --- group 2: fusion / hypergraph ---
    'no-vue',             # e = Linear(concat(a,t,v)) instead of attention fusion
    'no-hypergraph',      # h_final = h_init, classifier concatenates the 4 raw nodes
    'no-ind-edges',       # keep only coup_at / coup_av / coup_tv
    'no-coup-edges',      # keep only ind_a / ind_t / ind_v
)

# Ablations whose TEACHER is provably the baseline teacher. Both priors
# below are gated by `availability`, which is all-ones when mask_rate=0:
#     w_R = availability + (1 - availability) * reliability = 1
# so the module they remove already contributes nothing to a teacher.
# Measured: |grad| = 0.0 for the VUE gate at mask_rate=0, 4.089 at 0.3.
# Pass --teacher-ablate none for these and skip retraining a teacher.
TEACHER_EQUIVALENT_TO_BASELINE = ('no-uncertainty',)


def teacher_ckpt_tag(ablate):
    """Filename suffix for a teacher trained under `ablate`.

    'none' returns '' so baseline checkpoints keep the path they already
    have on disk -- this is what makes the change backward compatible.
    """
    if ablate not in ABLATE_CHOICES:
        raise ValueError(f'unknown ablate={ablate!r}; choose from {ABLATE_CHOICES}')
    return '' if ablate == 'none' else f'_ablate-{ablate}'


def add_ablate_args(parser, student=False):
    """Add --ablate (and, for a student, --teacher-ablate)."""
    parser.add_argument(
        '--ablate', type=str, default='none', choices=ABLATE_CHOICES,
        help='Remove one module or prior. Loss-cluster ablations use '
             '--lambda-prob/-proto/-rel 0 instead and are not listed here.')
    if student:
        parser.add_argument(
            '--teacher-ablate', type=str, default=None, choices=ABLATE_CHOICES,
            help='Which teacher checkpoint to distil from. Defaults to --ablate, '
                 'which is what an architecture ablation needs: student and teacher '
                 f'must match. Set to none for {TEACHER_EQUIVALENT_TO_BASELINE}, '
                 'where the ablated teacher is provably the baseline teacher.')


def resolve_teacher_ablate(args):
    """--teacher-ablate defaults to --ablate; warn on a deliberate mismatch."""
    ablate = getattr(args, 'ablate', 'none')
    teacher_ablate = getattr(args, 'teacher_ablate', None)
    if teacher_ablate is None:
        teacher_ablate = ablate
    if teacher_ablate != ablate:
        print(f'  NOTE: student ablate={ablate} distilling from a '
              f'teacher trained with ablate={teacher_ablate}.')
        if ablate not in TEACHER_EQUIVALENT_TO_BASELINE or teacher_ablate != 'none':
            print('  WARNING: that pairing is not one of the provably-equivalent '
                  f'cases {TEACHER_EQUIVALENT_TO_BASELINE} -> none. The state_dict '
                  'will still load (shapes match across the grid), so this will '
                  'NOT raise -- check it is what you meant.')
    return teacher_ablate
