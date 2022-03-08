#ifndef DEEP_CFR_LOCAL_BEST_RESPONSE
#define DEEP_CFR_LOCAL_BEST_RESPONSE

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/algorithm/container.h"
#include "open_spiel/algorithms/public_tree.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/barrier.h"
#include "open_spiel/utils/thread_pool.h"
#include "vpnet.h"

namespace open_spiel {
namespace universal_poker {

class LocalBestResponse {
  // Only for universal poker with 2 players, and 2 rounds.
 public:
  LocalBestResponse(const Game &game, const Policy &policy,
                    algorithms::CFRNetModel *net = nullptr,
                    int num_threads = 1);

  ~LocalBestResponse() = default;

  std::pair<double, double> operator()(int batch_size, bool verbose = false);

  Eigen::ArrayXd br_run(Player player, int batch_size);

 private:
  Eigen::ArrayXd enter_br(Player player, algorithms::PublicNode *node,
                          const Eigen::ArrayXd &p_s_ni,
                          const logic::CardSet &outcome, int batch_size,
                          int index);

  logic::CardSet _deal_cards(int start_round, int end_round,
                             const logic::CardSet &hole_cards,
                             const logic::CardSet &outcome, std::mt19937 *rng);

  double rollout(Player player, algorithms::PublicNode *node,
                 const Eigen::ArrayXd &p_s_ni, const logic::CardSet &hole_cards,
                 const logic::CardSet &outcome, int index, std::mt19937 *rng);

  double Evaluate(Player player, const UniversalPokerState *state,
                  const Eigen::ArrayXd &q, const logic::CardSet &hole_cards,
                  const logic::CardSet &outcome, int index);

  double _dist_br_chance(Player player, algorithms::PublicNode *node,
                         const Eigen::ArrayXd &p_s_ni,
                         const logic::CardSet &hole_cards,
                         const logic::CardSet &outcome, int index,
                         std::mt19937 *rng);

  double _br_recursive(Player player, algorithms::PublicNode *node,
                       const Eigen::ArrayXd &p_s_ni,
                       const logic::CardSet &hole_cards,
                       const logic::CardSet &outcome, int index, bool lookahead,
                       std::mt19937 *rng);

 private:
  int step_;
  const UniversalPokerGame *game_;
  const Policy *policy_;
  algorithms::CFRNetModel *net_;
  std::mt19937 *rng_;
  std::vector<algorithms::PublicTree> trees_;
  const acpc_cpp::ACPCGame *acpc_game_;
  logic::CardSet deck_;
  std::vector<logic::CardSet> player_outcomes_;
  std::vector<std::vector<uint8_t>> player_outcome_arrays_;
  int num_outcomes_;
  int num_cards_;
  int num_hole_cards_;
  int num_board_cards_;
  logic::CardSet default_cards_;

  int num_threads_;
  std::unique_ptr<ThreadPool> pool_;

  int over_ratio_;
};

}  // namespace universal_poker
}  // namespace open_spiel

#endif  // DEEP_CFR_LOCAL_BEST_RESPONSE