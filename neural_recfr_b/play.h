#include "deep_cfr.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/thread.h"
#include "vpevaluator.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {
void play(const Game &game, const DeepCFRConfig &config, Policy &policy,
          StopToken *stop);
}  // namespace algorithms
}  // namespace open_spiel