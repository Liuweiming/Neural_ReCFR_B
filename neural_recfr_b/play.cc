
#include <assert.h>
#include <getopt.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "deep_cfr.h"
#include "device_manager.h"
#include "open_spiel/algorithms/cfr.h"
#include "open_spiel/games/universal_poker.h"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/game.h"
#include "open_spiel/games/universal_poker/acpc/project_acpc_server/net.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/barrier.h"
#include "open_spiel/utils/circular_buffer.h"
#include "open_spiel/utils/data_logger.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/json.h"
#include "open_spiel/utils/logger.h"
#include "open_spiel/utils/lru_cache.h"
#include "open_spiel/utils/reservior_buffer.h"
#include "open_spiel/utils/stats.h"
#include "open_spiel/utils/thread.h"
#include "open_spiel/utils/threaded_queue.h"
#include "vpevaluator.h"
#include "vpnet.h"

namespace open_spiel {
namespace algorithms {
void play(const Game &game, const DeepCFRConfig &config, Policy &policy,
          StopToken *stop) {
  FileLogger logger(config.path, absl::StrCat("player", "-mpi-", 0));
  std::random_device rd;
  std::mt19937 rng(rd());
  omp_set_num_threads(config.omp_threads);

  std::unique_ptr<PublicTree> tree;
  tree.reset(new PublicTree(game.NewInitialState()));
  PublicNode *root_node = tree->Root();
  universal_poker::UniversalPokerState *root_state =
      static_cast<universal_poker::UniversalPokerState *>(
          root_node->GetState());

  const universal_poker::acpc_cpp::ACPCGame *acpc_game =
      root_state->GetACPCGame();
  const universal_poker::acpc_cpp::ACPCState *root_acpc_state =
      root_state->GetACPCState();

  std::unordered_map<std::string, std::string> game_defs = {
      {"kuhn_poker", "kuhn.limit.2p.game"},
      {"leduc_poker", "leduc.limit.2p.game"},
      {"FHP_poker", "FHP.limit.2p.game"},
      {"HULH_poker", "holdem.limit.2p.game"}};
  std::string game_def = game_defs[config.game];

  // codes imported from project_acpc_server/exmpale_player.c
  int sock, len, r, a;
  int32_t min, max;
  std::string host_name = config.host;
  uint16_t port = config.port;
  double p;
  project_acpc_server::Game *raw_game;
  project_acpc_server::MatchState raw_state;
  project_acpc_server::Action action;
  FILE *file, *toServer, *fromServer;
  struct timeval tv;
  double probs[NUM_ACTION_TYPES];
  double actionProbs[NUM_ACTION_TYPES];
  char line[MAX_LINE_LEN];

  /* we make some assumptions about the actions - check them here */
  assert(NUM_ACTION_TYPES == 3);

  /* get the game */
  file = fopen(game_def.c_str(), "r");
  if (file == NULL) {
    fprintf(stderr, "ERROR: could not open game %s\n", game_def.c_str());
    exit(EXIT_FAILURE);
  }
  raw_game = project_acpc_server::readGame(file);
  if (raw_game == NULL) {
    fprintf(stderr, "ERROR: could not read game %s\n", game_def.c_str());
    exit(EXIT_FAILURE);
  }
  fclose(file);
  char host_name_c[100];
  strncpy(host_name_c, host_name.c_str(), host_name.size());
  host_name_c[host_name.size()] = '\0';
  sock = project_acpc_server::connectTo(host_name_c, port);
  if (sock < 0) {
    exit(EXIT_FAILURE);
  }
  toServer = fdopen(sock, "w");
  fromServer = fdopen(sock, "r");
  if (toServer == NULL || fromServer == NULL) {
    fprintf(stderr, "ERROR: could not get socket streams\n");
    exit(EXIT_FAILURE);
  }

  /* send version string to dealer */
  if (fprintf(toServer, "VERSION:%" PRIu32 ".%" PRIu32 ".%" PRIu32 "\n",
              VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION) != 14) {
    fprintf(stderr, "ERROR: could not get send version to server\n");
    exit(EXIT_FAILURE);
  }
  fflush(toServer);

  std::string last_actions;
  /* play the game! */
  while (fgets(line, MAX_LINE_LEN, fromServer)) {
    /* ignore comments */
    if (line[0] == '#' || line[0] == ';') {
      continue;
    }

    len = readMatchState(line, raw_game, &raw_state);
    if (len < 0) {
      fprintf(stderr, "ERROR: could not read state %s", line);
      exit(EXIT_FAILURE);
    }

    if (stateFinished(&raw_state.state)) {
      /* ignore the game over message */

      continue;
    }

    if (currentPlayer(raw_game, &raw_state.state) != raw_state.viewingPlayer) {
      /* we're not acting */

      continue;
    }

    /* add a colon (guaranteed to fit because we read a new-line in fgets)
     */
    line[len] = ':';
    ++len;

    uint8_t current_player = raw_state.viewingPlayer;
    universal_poker::acpc_cpp::ACPCState acpc_state(
        acpc_game, *static_cast<universal_poker::acpc_cpp::RawACPCState *>(
                       &raw_state.state));
    std::vector<uint8_t> hole_cards = acpc_state.HoleCards(current_player);
    std::vector<uint8_t> board_cards =
        acpc_state.BoardCards(acpc_state.SumBoardCards());
    std::string sequence;
    std::vector<std::string> seqs;
    for (int r = 0; r <= acpc_state.GetRound(); ++r) {
      std::string bet = acpc_state.BettingSequence(r);
      if (bet.size()) {
        seqs.push_back(bet);
      }
    }
    sequence = absl::StrJoin(seqs, ",");
    PublicNode *current_node = tree->GetByHistory(sequence);
    universal_poker::UniversalPokerState *current_state =
        static_cast<universal_poker::UniversalPokerState *>(
            current_node->GetState());
    current_state->SetHoleCards(current_player, hole_cards);
    current_state->SetBoardCards(board_cards);
    std::vector<Action> legal_actions = current_state->LegalActions();
    ActionsAndProbs ap = policy.GetStatePolicy(*current_state);
    auto action_proba = SampleAction(ap, rng);
    std::cout << ap << std::endl;
    Action action = action_proba.first;
    SPIEL_CHECK_TRUE(std::find(legal_actions.begin(), legal_actions.end(),
                               action) != legal_actions.end());
    std::cout << absl::StrFormat(
                     "histroy: %-20s info: %-20s action: %-10s proba: %-3f",
                     sequence,
                     current_state->InformationStateString(current_player),
                     current_state->ActionToString(current_player, action),
                     action_proba.second)
              << std::endl;

    /* do the action! */
    std::string action_str =
        current_state->ActionToString(current_player, action);
    for (int c_i = 0; c_i != action_str.size(); ++c_i) {
      line[len] = action_str[c_i];
      ++len;
    }
    line[len] = '\r';
    ++len;
    line[len] = '\n';
    ++len;
    line[len] = '\0';
    std::cout << len << " " << line << std::endl;
    if (fwrite(line, 1, len, toServer) != len) {
      fprintf(stderr, "ERROR: could not get send response to server\n");
      exit(EXIT_FAILURE);
    }
    fflush(toServer);
  }
}
}  // namespace algorithms
}  // namespace open_spiel
