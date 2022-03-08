#include "neural_recfr_b_solver.h"

#include <memory>
#include <numeric>
#include <random>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {

std::ostream& operator<<(std::ostream& out, const ReplayNode& node) {
  out << node.information << " " << node.current_player << " "
      << node.legal_actions << " " << node.action_index << " "
      << node.next_information << " " << node.next_legal_actions << " "
      << node.next_action_index << " " << node.value << " " << node.player_reach
      << " " << node.opponent_reach << " " << node.sampling_reach;
  return out;
}

NeuralReCFRBSolver::NeuralReCFRBSolver(
    const Game& game, std::shared_ptr<VPNetEvaluator> value_0_eval,
    std::shared_ptr<VPNetEvaluator> value_1_eval,
    std::shared_ptr<VPNetEvaluator> policy_0_eval,
    std::shared_ptr<VPNetEvaluator> policy_1_eval, bool use_regret_net,
    bool use_policy_net, bool use_tabular, bool anticipatory, double alpha,
    double eta, double epsilon, bool symmetry, std::mt19937* rng,
    AverageType avg_type)
    : game_(game.Clone()),
      rng_(rng),
      iterations_(0),
      avg_type_(avg_type),
      dist_(0.0, 1.0),
      value_eval_{value_0_eval, value_1_eval},
      policy_eval_{policy_0_eval, policy_1_eval},
      tree_(game_->NewInitialState()),
      root_node_(tree_.Root()),
      root_state_(root_node_->GetState()),
      use_regret_net(use_regret_net),
      use_policy_net(use_policy_net),
      use_tabular(use_tabular),
      anticipatory_(anticipatory),
      alpha_(alpha),
      eta_(eta),
      epsilon_(epsilon),
      symmetry_(symmetry) {
  if (game_->GetType().dynamics != GameType::Dynamics::kSequential) {
    SpielFatalError(
        "MCCFR requires sequential games. If you're trying to run it "
        "on a simultaneous (or normal-form) game, please first transform it "
        "using turn_based_simultaneous_game.");
  }
}
std::pair<std::vector<Trajectory>, std::vector<Trajectory>>
NeuralReCFRBSolver::RunIteration() {
  std::vector<Trajectory> value_trajectories(game_->NumPlayers());
  std::vector<Trajectory> policy_trajectories(game_->NumPlayers());
  for (auto p = Player{0}; p < game_->NumPlayers(); ++p) {
    auto ret_p = RunIteration(rng_, 1e-5, p, 0);
    value_trajectories.push_back(ret_p.first);
    policy_trajectories.push_back(ret_p.second);
  }
  return {value_trajectories, policy_trajectories};
}

std::pair<Trajectory, Trajectory> NeuralReCFRBSolver::RunIteration(
    Player player, double alpha, int step) {
  return RunIteration(rng_, player, alpha, step);
}

std::pair<Trajectory, Trajectory> NeuralReCFRBSolver::RunIteration(
    std::mt19937* rng, Player player, double alpha, int step) {
  alpha_ = alpha;
  node_touch_ = 0;
  ++iterations_;
  Trajectory value_trajectory;
  Trajectory policy_trajectory;
  // Sample a chace seed at the start of an iteration.
  ChanceData chance_data = root_state_->SampleChance(rng);
  bool current_or_average[2] = {true, true};
  if (anticipatory_) {
    current_or_average[player] = dist_(*rng) < eta_;
    current_or_average[player == 0 ? 1 : 0] = dist_(*rng) < eta_;
  }
  // NOTE: We do not need to clearCache if the networks are never updated. So
  // the Cache should be clear by the learner. Don't do this:
  // value_eval_->ClearCache();
  UpdateRegrets(root_node_, player, 1, 1, 1, 1, value_trajectory,
                policy_trajectory, step, rng, chance_data, current_or_average);
  return {value_trajectory, policy_trajectory};
}

double _backward_update_regret(double alpha, double opponent_reach,
                               const std::vector<double>& values, int step) {
  int num_actions = values.size();
  double threshold = alpha * opponent_reach / step;
  std::vector<double> sorted_value = values;
  std::sort(sorted_value.begin(), sorted_value.end());
  std::vector<double> available_values(num_actions);
  auto global_min_value = sorted_value[0] - sqrt(threshold);
  std::vector<double> ubound(num_actions, 0.0);
  for (int i = 0; i != num_actions; ++i) {
    for (int j = i; j != num_actions; ++j) {
      ubound[i] += std::pow(sorted_value[j] - sorted_value[i], 2);
    }
  }
  for (int i = 0; i != num_actions; ++i) {
    if (threshold <= ubound[i]) {
      continue;
    }
    double a = num_actions - i;
    double b = 0;
    double c = -threshold;
    for (int j = i; j != num_actions; ++j) {
      b += -2 * sorted_value[j];
      c += std::pow(sorted_value[j], 2);
    }
    auto x1 = (-b - sqrt(std::max(0.0, b * b - 4 * a * c))) / (2 * a);
    return x1;
  }
  return 0.0;
}

double NeuralReCFRBSolver::UpdateRegrets(
    PublicNode* node, Player player, double player_reach, double opponent_reach,
    double ave_opponent_reach, double sampling_reach,
    Trajectory& value_trajectory, Trajectory& policy_trajectory, int step,
    std::mt19937* rng, const ChanceData& chance_data,
    bool current_or_average[2]) {
  State& state = *(node->GetState());
  universal_poker::UniversalPokerState* poker_state =
      static_cast<universal_poker::UniversalPokerState*>(node->GetState());
  state.SetChance(chance_data);
  // std::cout << state.ToString() << std::endl;
  if (state.IsTerminal()) {
    double value = state.PlayerReturn(player);
    value_trajectory.states.push_back(ReplayNode{
        state.InformationStateString(player),
        state.InformationStateTensor(player), player, std::vector<Action>{}, -1,
        std::vector<double>{}, std::vector<Action>{}, -1, value, 1.0,
        player_reach, opponent_reach, sampling_reach});
    return value;
  } else if (state.IsChanceNode()) {
    Action action = SampleAction(state.ChanceOutcomes(), dist_(*rng)).first;
    return UpdateRegrets(node->GetChild(action), player, player_reach,
                         opponent_reach, ave_opponent_reach, sampling_reach,
                         value_trajectory, policy_trajectory, step, rng,
                         chance_data, current_or_average);
  } else if (state.IsSimultaneousNode()) {
    SpielFatalError(
        "Simultaneous moves not supported. Use "
        "TurnBasedSimultaneousGame to convert the game first.");
  }

  node_touch_ += 1;

  Player cur_player = state.CurrentPlayer();
  std::string is_key = state.InformationStateString(cur_player);
  std::vector<Action> legal_actions = state.LegalActions();
  std::vector<double> information_tensor = state.InformationStateTensor();

  // NOTE: why we need a copy here? don't copy, just create one.
  CFRInfoStateValues info_state_copy(legal_actions, kInitialTableValues);
  CFRInfoStateValues current_info_state(legal_actions, kInitialTableValues);
  if (step != 1) {
    auto inf_output = value_eval_[cur_player]->Inference(state).value;
    // std::cout << "inf: " << inf_output << std::endl;
    SPIEL_CHECK_EQ(inf_output.size(), legal_actions.size());
    std::vector<double> cfr_regret = inf_output;
    double delta = poker_state->MaxUtility() - poker_state->MinUtility();
    double cfr_value =
        _backward_update_regret(alpha_,
                                delta * delta * legal_actions.size() /
                                    pow(ave_opponent_reach + 1e-7, 1),
                                cfr_regret, step);
    for (int i = 0; i != legal_actions.size(); ++i) {
      cfr_regret[i] -= cfr_value;
    }
    current_info_state.SetRegret(cfr_regret);
  }
  current_info_state.ApplyRegretMatchingUsingMax();
  if (step != 1 && (symmetry_ || cur_player != player)) {
    CFRNetModel::InferenceInputs inference_input{is_key, legal_actions,
                                                 information_tensor};
    auto cfr_policy = policy_eval_[cur_player]->Inference(state).value;
    double eta = 1.0 / step;
    for (auto& p : cfr_policy) {
      p = eta * 1.0 / cfr_policy.size() + (1 - eta) * p;
    }
    current_info_state.SetCumulatePolicy(cfr_policy);
  }
  if (current_or_average[cur_player]) {
    info_state_copy = current_info_state;
  } else {
    if (step != 1 && (symmetry_ || cur_player != player)) {
      info_state_copy.SetPolicy(current_info_state.cumulative_policy);
    } else {
      std::vector<double> uniform_strategy(legal_actions.size(),
                                           1.0 / legal_actions.size());
      info_state_copy.SetPolicy(uniform_strategy);
    }
  }
  double value = 0;
  int aidx = 0;
  if (cur_player == player) {
    aidx = info_state_copy.SampleActionIndex(0.0, dist_(*rng));
    double new_reach = current_info_state.current_policy[aidx] * player_reach;
    double new_sampling_reach =
        info_state_copy.current_policy[aidx] * sampling_reach;
    value = UpdateRegrets(
        node->GetChild(legal_actions[aidx]), player, new_reach, opponent_reach,
        ave_opponent_reach, new_sampling_reach, value_trajectory,
        policy_trajectory, step, rng, chance_data, current_or_average);
  } else {
    aidx = info_state_copy.SampleActionIndex(0.0, dist_(*rng));
    double new_reach = current_info_state.current_policy[aidx] * opponent_reach;
    double new_ave_reach = 1.0 / legal_actions.size() * ave_opponent_reach;
    double new_sampling_reach =
        info_state_copy.current_policy[aidx] * sampling_reach;
    value = UpdateRegrets(
        node->GetChild(legal_actions[aidx]), player, player_reach, new_reach,
        new_ave_reach, new_sampling_reach, value_trajectory, policy_trajectory,
        step, rng, chance_data, current_or_average);
  }

  if (cur_player == player) {
    double delta = poker_state->MaxUtility() - poker_state->MinUtility();
    value_trajectory.states.push_back(ReplayNode{
        is_key, information_tensor, cur_player, legal_actions, aidx,
        std::vector<double>{}, std::vector<Action>{}, -1, 0, 1.0, player_reach,
        delta * delta * legal_actions.size() /
            pow(ave_opponent_reach + 1e-7, 1),
        sampling_reach});
  }

  if (cur_player == player && current_or_average[cur_player]) {
    if (!use_policy_net || use_tabular) {
      std::vector<double> policy(legal_actions.size());
      for (int paidx = 0; paidx < legal_actions.size(); ++paidx) {
        policy[paidx] = current_info_state.current_policy[paidx] *
                        player_reach / sampling_reach;
      }
      policy_eval_[cur_player]->AccumulateCFRTabular(state, policy);
    }
    if (use_policy_net) {
      policy_trajectory.states.push_back(
          ReplayNode{is_key, information_tensor, cur_player, legal_actions,
                     aidx, std::vector<double>{}, std::vector<Action>{}, -1, 1,
                     1.0, player_reach, opponent_reach, sampling_reach});
    }
  }
  return value;
}  // namespace algorithms

}  // namespace algorithms
}  // namespace open_spiel
