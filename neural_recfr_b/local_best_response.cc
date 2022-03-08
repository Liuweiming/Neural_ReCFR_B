#include "local_best_response.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/barrier.h"

namespace open_spiel {
namespace universal_poker {
using namespace open_spiel::algorithms;

namespace {
long long C(int n, int r) {
  if (r > n - r) r = n - r;
  long long ans = 1;
  int i;

  for (i = 1; i <= r; i++) {
    ans *= n - r + i;
    ans /= i;
  }
  return ans;
}
}  // namespace

LocalBestResponse::LocalBestResponse(const Game &game, const Policy &policy,
                                     CFRNetModel *net, int num_threads)
    : game_(static_cast<const UniversalPokerGame *>(&game)),
      policy_(&policy),
      net_(net),
      num_threads_(num_threads),
      pool_(new ThreadPool(num_threads)),
      over_ratio_(1),
      acpc_game_(game_->GetACPCGame()),
      deck_(/*num_suits=*/acpc_game_->NumSuitsDeck(),
            /*num_ranks=*/acpc_game_->NumRanksDeck()) {
  num_hole_cards_ = acpc_game_->GetNbHoleCardsRequired();
  num_board_cards_ = acpc_game_->GetTotalNbBoardCards();
  player_outcomes_ = deck_.SampleCards(num_hole_cards_);
  player_outcome_arrays_.resize(player_outcomes_.size());
  absl::c_transform(player_outcomes_, player_outcome_arrays_.begin(),
                    [](const logic::CardSet &cs) { return cs.ToCardArray(); });

  num_cards_ = acpc_game_->NumSuitsDeck() * acpc_game_->NumRanksDeck();

  int num_trees = (num_threads > 1) ? (over_ratio_ * num_threads) : 1;
  for (int i = 0; i < num_trees; ++i) {
    trees_.emplace_back(game_->NewInitialState());
  }
}

std::pair<double, double> LocalBestResponse::operator()(int batch_size,
                                                        bool verbose) {
  auto br_0 = br_run(0, batch_size);
  auto br_1 = br_run(1, batch_size);
  auto exp = (br_0 + br_1) / 2;
  double mean = exp.mean();
  double dv = std::sqrt((exp - mean).square().sum() / (exp.size() - 1));
  if (verbose) {
    std::stringstream exp_ss;
    for (double e : exp) {
      exp_ss << "LBR: " << e << "\n";
    }
    std::cout << exp_ss.str() << std::endl;
  }
  return {mean, dv / std::sqrt(batch_size)};
}

Eigen::ArrayXd LocalBestResponse::br_run(Player player, int batch_size) {
  step_ = 0;
  Eigen::ArrayXd p_s_ni = Eigen::ArrayXd::Ones(player_outcomes_.size());
  Eigen::ArrayXd values = Eigen::ArrayXd::Zero(batch_size);
  SPIEL_CHECK_GE(batch_size, num_threads_);
  int batch_per_thread = batch_size / num_threads_;
  int rest = batch_size - batch_per_thread * num_threads_;
  std::vector<std::future<Eigen::ArrayXd>> ret;
  for (int t = 0; t != num_threads_; ++t) {
    int bt = batch_per_thread;
    if (t < rest) {
      bt += 1;
    }
    ret.emplace_back(pool_->enqueue(&LocalBestResponse::enter_br, this, player,
                                    trees_[t].Root(), p_s_ni, default_cards_,
                                    bt, t));
  }
  double value = 0;
  int batch_start = 0;
  for (int t = 0; t != num_threads_; ++t) {
    int bt = batch_per_thread;
    if (t < rest) {
      bt += 1;
    }
    auto sub_values = ret[t].get();
    values.segment(batch_start, bt) = sub_values;
    value += sub_values.sum();
    batch_start += bt;
  }
  return values;
}

Eigen::ArrayXd LocalBestResponse::enter_br(Player player,
                                           algorithms::PublicNode *node,
                                           const Eigen::ArrayXd &p_s_ni,
                                           const logic::CardSet &outcome,
                                           int batch_size, int index) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> rnd(0, player_outcomes_.size() - 1);
  Eigen::ArrayXd values = Eigen::ArrayXd::Zero(batch_size);
  for (int b = 0; b != batch_size; ++b) {
    int hole_index = rnd(rng);
    Eigen::ArrayXd new_p_s_ni = p_s_ni;
    logic::CardSet hole_cards = player_outcomes_[hole_index];
    int check_sum = 2 * num_hole_cards_;
    for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
      logic::CardSet check_cards = hole_cards;
      check_cards.Combine(player_outcomes_[inf_id]);
      if (check_cards.NumCards() != check_sum) {
        new_p_s_ni(inf_id) = 0;
      }
    }
    values[b] = _br_recursive(player, node, new_p_s_ni, hole_cards, outcome,
                              index, false, &rng);
  }
  return values;
}

static uint8_t DealCard(uint8_t *deck, const int numCards, std::mt19937 *rng,
                        std::uniform_int_distribution<int> &rand_int) {
  int i;
  uint8_t ret;
  SPIEL_CHECK_GE(numCards, 1);
  i = rand_int(*rng,
               std::uniform_int_distribution<int>::param_type{0, numCards - 1});
  ret = deck[i];
  deck[i] = deck[numCards - 1];

  return ret;
}

logic::CardSet LocalBestResponse::_deal_cards(int start_round, int end_round,
                                              const logic::CardSet &hole_cards,
                                              const logic::CardSet &outcome,
                                              std::mt19937 *rng) {
  std::uniform_int_distribution<int> rand_int;
  logic::CardSet deck_r = deck_;
  std::vector<uint8_t> hole_cards_vec = hole_cards.ToCardArray();
  std::vector<uint8_t> outcome_vec = outcome.ToCardArray();
  for (auto &c : hole_cards_vec) {
    deck_r.RemoveCard(c);
  }
  for (auto &c : outcome_vec) {
    deck_r.RemoveCard(c);
  }
  std::vector<uint8_t> deck_r_vec = deck_r.ToCardArray();
  int num_deck = deck_r_vec.size();
  int required_cards = acpc_game_->GetNbBoardCardsRequired(end_round) -
                       acpc_game_->GetNbBoardCardsRequired(start_round - 1);
  std::vector<int> rest_cards(required_cards, 0);
  int offset = 0;
  for (int r = start_round; r <= end_round; ++r) {
    for (int i = 0; i < acpc_game_->GetNbBoardCardsAtRound(r); ++i) {
      SPIEL_CHECK_LE(offset + i, required_cards);
      rest_cards[offset + i] =
          DealCard(&(deck_r_vec[0]), num_deck, rng, rand_int);
      --num_deck;
    }
    std::sort(
        rest_cards.begin() + offset,
        rest_cards.begin() + offset + acpc_game_->GetNbBoardCardsAtRound(r));
    offset += acpc_game_->GetNbBoardCardsAtRound(r);
  }
  logic::CardSet rest_card_set(rest_cards);
  logic::CardSet new_outcome = outcome;
  new_outcome.Combine(rest_card_set);
  SPIEL_CHECK_EQ(new_outcome.NumCards(),
                 acpc_game_->GetNbBoardCardsRequired(end_round));
  return new_outcome;
}

double LocalBestResponse::rollout(Player player, algorithms::PublicNode *node,
                                  const Eigen::ArrayXd &p_s_ni,
                                  const logic::CardSet &hole_cards,
                                  const logic::CardSet &outcome, int index,
                                  std::mt19937 *rng) {
  UniversalPokerState *state =
      static_cast<UniversalPokerState *>(node->GetState());
  if (state->IsTerminal()) {
    return Evaluate(player, state, p_s_ni, hole_cards, outcome, index);
  }
  double value = 0;
  int rest_num_cards = num_board_cards_ - outcome.NumCards();
  std::vector<logic::CardSet> card_sets_to_roll;
  if (rest_num_cards == 0) {
    card_sets_to_roll.push_back(outcome);
  } else if (rest_num_cards <= 1) {
    logic::CardSet deck = deck_;
    for (auto c : hole_cards.ToCardArray()) {
      deck.RemoveCard(c);
    }
    for (auto c : outcome.ToCardArray()) {
      deck.RemoveCard(c);
    }
    card_sets_to_roll = deck.SampleCards(rest_num_cards);
    for (int b = 0; b != card_sets_to_roll.size(); ++b) {
      card_sets_to_roll[b].Combine(outcome);
    }
  } else {
    int rollout_batch_size = C(num_cards_, 1);
    int round = state->GetACPCState()->GetRound();
    int start_round = round;
    if (acpc_game_->GetNbBoardCardsRequired(round) == outcome.NumCards()) {
      start_round += 1;
    }
    for (int b = 0; b != rollout_batch_size; ++b) {
      // deal rest cards.
      logic::CardSet new_outcome = _deal_cards(
          start_round, acpc_game_->NumRounds() - 1, hole_cards, outcome, rng);

      card_sets_to_roll.push_back(new_outcome);
    }
  }
  for (int b = 0; b != card_sets_to_roll.size(); ++b) {
    // deal rest cards.
    logic::CardSet new_outcome = card_sets_to_roll[b];
    Eigen::ArrayXd new_p_s_ni = p_s_ni;
    // set oppo reach.
    for (int i = 0; i < player_outcomes_.size(); ++i) {
      logic::CardSet cs = player_outcomes_[i];
      cs.Combine(new_outcome);
      if (cs.NumCards() != (num_hole_cards_ + num_board_cards_)) {
        new_p_s_ni[i] = 0;
      }
    }
    value +=
        Evaluate(player, state, new_p_s_ni, hole_cards, new_outcome, index);
  }
  return value / card_sets_to_roll.size();
}

double LocalBestResponse::Evaluate(Player player,
                                   const UniversalPokerState *state,
                                   const Eigen::ArrayXd &q,
                                   const logic::CardSet &hole_cards,
                                   const logic::CardSet &outcome, int index) {
  // values = p * (value_matrx * valid_matrix_) \dot q.
  // For two players.
  double player_spent = state->acpc_state_.CurrentSpent(player);
  double other_spent =
      state->acpc_state_.TotalSpent() - state->acpc_state_.CurrentSpent(player);
  int num_dealt = hole_cards.NumCards() + outcome.NumCards();
  double hole_proba = 1.0 / C(num_cards_ - num_dealt, num_hole_cards_);
  if (state->acpc_state_.NumFolded() >= state->acpc_game_->GetNbPlayers() - 1) {
    // Some one folded here.
    double scale_value = 0;
    if (state->acpc_state_.PlayerFolded(player)) {
      scale_value = -player_spent;
    } else {
      scale_value = other_spent;
    }
    return hole_proba * scale_value * q.sum();
  } else {
    Eigen::ArrayXd compared(player_outcomes_.size());
    logic::CardSet player_hand = hole_cards;
    player_hand.Combine(outcome);
    SPIEL_CHECK_EQ(outcome.NumCards(), num_board_cards_);
    SPIEL_CHECK_EQ(player_hand.NumCards(),
                   (num_hole_cards_ + num_board_cards_));
    int player_rank = player_hand.RankCards();
    for (int i = 0; i < player_outcomes_.size(); ++i) {
      logic::CardSet cs = player_outcomes_[i];
      cs.Combine(outcome);
      int oppo_rank;
      if (cs.NumCards() != (num_hole_cards_ + num_board_cards_)) {
        oppo_rank = -1;
      } else {
        oppo_rank = cs.RankCards();
      }
      compared[i] =
          (player_rank > oppo_rank) ? 1 : ((player_rank < oppo_rank) ? -1 : 0);
    }
    return player_spent * hole_proba * (compared * q).sum();
  }
}

double LocalBestResponse::_dist_br_chance(Player player,
                                          algorithms::PublicNode *node,
                                          const Eigen::ArrayXd &p_s_ni,
                                          const logic::CardSet &hole_cards,
                                          const logic::CardSet &outcome,
                                          int index, std::mt19937 *rng) {
  UniversalPokerState *state =
      static_cast<UniversalPokerState *>(node->GetState());
  int round = state->GetACPCState()->GetRound();
  int required_cards = acpc_game_->GetNbBoardCardsRequired(round);
  SPIEL_CHECK_GT(round, 0);
  SPIEL_CHECK_GT(required_cards, 0);
  SPIEL_CHECK_LE(outcome.NumCards(), required_cards);
  logic::CardSet new_outcome =
      _deal_cards(round, round, hole_cards, outcome, rng);
  int check_num = num_hole_cards_ + new_outcome.NumCards();
  Eigen::ArrayXd new_p_s_ni = p_s_ni;
  for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
    logic::CardSet check_cards = new_outcome;
    check_cards.Combine(player_outcomes_[inf_id]);
    if (check_cards.NumCards() != check_num) {
      new_p_s_ni(inf_id) = 0;
    }
  }
  algorithms::PublicNode *new_node = node->GetChild(node->GetChildActions()[0]);
  double value = _br_recursive(player, new_node, new_p_s_ni, hole_cards,
                               new_outcome, index, false, rng);
  return value;
}

double LocalBestResponse::_br_recursive(
    Player player, algorithms::PublicNode *node, const Eigen::ArrayXd &p_s_ni,
    const logic::CardSet &hole_cards, const logic::CardSet &outcome, int index,
    bool lookahead, std::mt19937 *rng) {
  UniversalPokerState *state =
      static_cast<UniversalPokerState *>(node->GetState());
  if (state->IsTerminal()) {
    return Evaluate(player, state, p_s_ni, hole_cards, outcome, index);
  }
  if (state->IsChanceNode()) {
    return _dist_br_chance(player, node, p_s_ni, hole_cards, outcome, index,
                           rng);
  }
  Player current_player = state->CurrentPlayer();
  std::vector<Action> legal_actions = state->LegalActions();
  if (current_player != player) {
    // dim 0: information index, dim 1: action index.
    Eigen::ArrayXXd s_sigma_o_a(player_outcomes_.size(), legal_actions.size());
    std::vector<CFRNetModel::InferenceInputs> net_inputs;
    for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
      if (p_s_ni(inf_id)) {
        // NOTE: We have to modify state, in order to get the information set.
        // So the _br_recursive must be not thread-saft.
        state->SetHoleCards(current_player, player_outcome_arrays_[inf_id]);
        state->SetBoardCards(outcome);
        if (net_ != nullptr) {
          net_inputs.push_back(CFRNetModel::InferenceInputs{
              node->GetState()->InformationStateString(), state->LegalActions(),
              node->GetState()->InformationStateTensor()});
        } else {
          auto inf_policy = policy_->GetStatePolicy(*state);
          for (int axid = 0; axid != legal_actions.size(); ++axid) {
            s_sigma_o_a(inf_id, axid) = inf_policy[axid].second;
          }
        }
      } else {
        s_sigma_o_a.row(inf_id) = 0;
      }
    }
    // NOTE: special inference when using cfrnet model.
    if (net_inputs.size() > 0) {
      std::vector<CFRNetModel::InferenceOutputs> net_outputs =
          net_->InfPolicy(current_player, net_inputs);
      int output_index = 0;
      for (int inf_id = 0; inf_id != player_outcomes_.size(); ++inf_id) {
        if (p_s_ni(inf_id)) {
          for (int axid = 0; axid != legal_actions.size(); ++axid) {
            s_sigma_o_a(inf_id, axid) = net_outputs[output_index].value[axid];
          }
          ++output_index;
        }
      }
    }

    double value = 0;
    if (lookahead) {
      SPIEL_CHECK_GE(legal_actions.size(), 2);
      for (int axid = 2; axid < legal_actions.size(); ++axid) {
        s_sigma_o_a.col(1) += s_sigma_o_a.col(axid);
      }
      for (int axid = 0; axid != 2; ++axid) {
        algorithms::PublicNode *next_node = node->GetChild(legal_actions[axid]);
        Eigen::ArrayXd new_p_s_ni = p_s_ni * s_sigma_o_a.col(axid);
        value += rollout(player, next_node, new_p_s_ni, hole_cards, outcome,
                         index, rng);
      }
      return value;
    }
    std::uniform_int_distribution<int> rnd(0, legal_actions.size() - 1);
    int axid = rnd(*rng);
    algorithms::PublicNode *next_node = node->GetChild(legal_actions[axid]);
    Eigen::ArrayXd new_p_s_ni =
        p_s_ni * s_sigma_o_a.col(axid) * legal_actions.size();
    value = _br_recursive(player, next_node, new_p_s_ni, hole_cards, outcome,
                          index, false, rng);
    return value;
  } else {
    // value of raise / bet
    std::vector<double> m(legal_actions.size());
    for (int axid = 0; axid != legal_actions.size(); ++axid) {
      algorithms::PublicNode *next_node = node->GetChild(legal_actions[axid]);
      UniversalPokerState *next_state =
          static_cast<UniversalPokerState *>(next_node->GetState());
      double player_spent = next_state->acpc_state_.CurrentSpent(player);
      double other_spent = next_state->acpc_state_.TotalSpent() -
                           next_state->acpc_state_.CurrentSpent(player);
      if (next_state->acpc_state_.PlayerFolded(player) ||
          player_spent == other_spent) {
        // fold or call
        m[axid] =
            rollout(player, next_node, p_s_ni, hole_cards, outcome, index, rng);
      } else {
        m[axid] = _br_recursive(player, next_node, p_s_ni, hole_cards, outcome,
                                index, true, rng);
      }
    }
    int max_axid = std::max_element(m.begin(), m.end()) - m.begin();
    algorithms::PublicNode *next_node = node->GetChild(legal_actions[max_axid]);
    return _br_recursive(player, next_node, p_s_ni, hole_cards, outcome, index,
                         false, rng);
  }
}
}  // namespace universal_poker
}  // namespace open_spiel