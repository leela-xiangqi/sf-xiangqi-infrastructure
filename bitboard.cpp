/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  modified for UCCI xiangqi engine
  Copyright (C) 2004-2008 Tord Romstad (Glaurung author)
  Copyright (C) 2008-2015 Marco Costalba, Joona Kiiski, Tord Romstad
  Copyright (C) 2015-2018 Marco Costalba, Joona Kiiski, Gary Linscott, Tord Romstad
  Copyright (C) 2018 Prcuvu

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>

#include "bitboard.h"

Bitboard::Bitboard()
{
    Bitboard(UINT64_C(0), UINT64_C(0));
}

Bitboard::Bitboard(const uint64_t high, const uint64_t low)
{
    value_u64[0] = low;
    value_u64[1] = high;
}

Bitboard::Bitboard(const int n)
{
    value_u64[0] = uint64_t(n);
    value_u64[1] = UINT64_C(0);
}

Bitboard::Bitboard(const Bitboard & source)
{
    _mm_store_si128(&value, source.value);
}

Bitboard::Bitboard(const __m128i & source)
{
    _mm_store_si128(&value, source);
}

Bitboard Bitboard::operator-(const Bitboard & source)
{
    return ((value_u64[0] < source.value_u64[0])
        ? Bitboard(value_u64[1] - source.value_u64[1] - 1, value_u64[0] - source.value_u64[0])
        : Bitboard(value_u64[1] - source.value_u64[1], value_u64[0] - source.value_u64[0])) &= AllSquares;
}

Bitboard Bitboard::operator-(const int n)
{
    return ((value_u64[0] < n)
        ? Bitboard(value_u64[1] - 1, value_u64[0] - n)
        : Bitboard(value_u64[1], value_u64[0] - n)) &= AllSquares;
}

Bitboard & Bitboard::operator=(const Bitboard & source)
{
    _mm_store_si128(&value, source.value);
    return *this;
}

Bitboard Bitboard::operator&(const Bitboard & source)
{
    return _mm_and_si128(value, source.value);
}

Bitboard & Bitboard::operator&=(const Bitboard & source)
{
    _mm_store_si128(&value, _mm_and_si128(value, source.value));
    return *this;
}

Bitboard Bitboard::operator|(const Bitboard & source)
{
    return _mm_or_si128(value, source.value);
}

Bitboard & Bitboard::operator|=(const Bitboard & source)
{
    _mm_store_si128(&value, _mm_or_si128(value, source.value));
    return *this;
}

Bitboard Bitboard::operator~() const
{
    return _mm_andnot_si128(value, AllSquares.value);
}

Bitboard Bitboard::operator^(const Bitboard & source)
{
    return _mm_xor_si128(value, source.value);
}

Bitboard & Bitboard::operator^=(const Bitboard & source)
{
    _mm_store_si128(&value, _mm_xor_si128(value, source.value));
    return *this;
}

Bitboard Bitboard::operator<<(const int n) const
{
    return Bitboard((n < 64) ? _mm_or_si128(_mm_srli_epi64(_mm_slli_si128(value, 8), 64 - n), _mm_slli_epi64(value, n))
                             : _mm_slli_epi64(_mm_slli_si128(value, 8), n - 64)) &= AllSquares;
}

/* Bitboard Bitboard::operator>>(const int n) const
{
    return (n < 64) ? _mm_or_si128(_mm_srli_epi64(value, n), _mm_slli_epi64(_mm_srli_si128(value, 8), 64 - n))
                    : _mm_srli_epi64(_mm_srli_si128(value, 8), n - 64);
} */

Bitboard::operator bool() const
{
    return (value_u64[0] | value_u64[1]) != UINT64_C(0);
}

uint8_t PopCnt16[1 << 16];
int SquareDistance[SQUARE_NB][SQUARE_NB];

Bitboard SquareBB[SQUARE_NB];
Bitboard FileBB[FILE_NB];
Bitboard RankBB[RANK_NB];
Bitboard AdjacentFilesBB[FILE_NB];
Bitboard ForwardRanksBB[COLOR_NB][RANK_NB];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
// Bitboard DistanceRingBB[SQUARE_NB][8];
Bitboard ForwardFileBB[COLOR_NB][SQUARE_NB];
// Bitboard PassedPawnMask[COLOR_NB][SQUARE_NB];
// Bitboard PawnAttackSpan[COLOR_NB][SQUARE_NB];
Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
Bitboard SoldierAttacks[COLOR_NB][SQUARE_NB];

Magic CannonMagics[SQUARE_NB];
Magic ChariotMagics[SQUARE_NB];
Magic HorseMagics[SQUARE_NB];
Magic ElephantMagics[SQUARE_NB];

namespace {

  Bitboard CannonTable[0xB40000];  // To store cannon attacks
  Bitboard ChariotTable[0x108000]; // To store chariot attacks
  Bitboard HorseTable[0x380];      // To store horse attacks
  Bitboard ElephantTable[0x9C];    // To store elephant attacks

  void init_magics(PieceType pt, Bitboard table[], Magic magics[]);

  // popcount16() counts the non-zero bits using SWAR-Popcount algorithm

  unsigned popcount16(unsigned u) {
    u -= (u >> 1) & 0x5555U;
    u = ((u >> 2) & 0x3333U) + (u & 0x3333U);
    u = ((u >> 4) + u) & 0x0F0FU;
    return (u * 0x0101U) >> 8;
  }
}


/// Bitboards::pretty() returns an ASCII representation of a bitboard suitable
/// to be printed to standard output. Useful for debugging.

const std::string Bitboards::pretty(Bitboard b) {

  std::string s = "+---+---+---+---+---+---+---+---+---+\n";

  for (Rank r = RANK_9; r >= RANK_0; --r)
  {
      for (File f = FILE_A; f <= FILE_I; ++f)
          s += b & make_square(f, r) ? "| X " : "|   ";

      s += "|\n+---+---+---+---+---+---+---+---+---+\n";
  }

  return s;
}


/// Bitboards::init() initializes various bitboard tables. It is called at
/// startup and relies on global objects to be already zero-initialized.

void Bitboards::init() {

  for (unsigned i = 0; i < (1 << 16); ++i)
      PopCnt16[i] = (uint8_t) popcount16(i);

  for (Square s = SQ_A0; s <= SQ_I9; ++s)
      SquareBB[s] = make_bitboard(s);

  for (File f = FILE_A; f <= FILE_I; ++f)
      FileBB[f] = f > FILE_A ? FileBB[f - 1] << 1 : FileABB;

  for (Rank r = RANK_0; r <= RANK_9; ++r)
      RankBB[r] = r > RANK_0 ? RankBB[r - 1] << 9 : Rank0BB;

  for (File f = FILE_A; f <= FILE_I; ++f)
      AdjacentFilesBB[f] = (f > FILE_A ? FileBB[f - 1] : 0) | (f < FILE_I ? FileBB[f + 1] : 0);

  for (Rank r = RANK_0; r < RANK_9; ++r)
      ForwardRanksBB[RED][r] = ~(ForwardRanksBB[BLACK][r + 1] = ForwardRanksBB[BLACK][r] | RankBB[r]);

  for (Color c = RED; c <= BLACK; ++c)
      for (Square s = SQ_A0; s <= SQ_I9; ++s)
      {
          ForwardFileBB [c][s] = ForwardRanksBB[c][rank_of(s)] & FileBB[file_of(s)];
          // PawnAttackSpan[c][s] = ForwardRanksBB[c][rank_of(s)] & AdjacentFilesBB[file_of(s)];
          // PassedPawnMask[c][s] = ForwardFileBB [c][s] | PawnAttackSpan[c][s];
      }

  for (Square s1 = SQ_A0; s1 <= SQ_I9; ++s1)
      for (Square s2 = SQ_A0; s2 <= SQ_I9; ++s2)
          if (s1 != s2)
          {
              SquareDistance[s1][s2] = distance<File>(s1, s2) + distance<Rank>(s1, s2);
              // DistanceRingBB[s1][SquareDistance[s1][s2] - 1] |= s2;
          }

  for (Color c = RED; c <= BLACK; ++c)
  {
      // Generate attacks for soldiers
      for (Square s = SQ_A0; s <= SQ_I9; ++s)
          for (int d : { NORTH, WEST, EAST })
          {
              Square to = s + Direction(c == RED ? d : -d);

              if (is_ok(to) && distance(s, to) == 1)
              {
                  SoldierAttacks[c][s] |= to;
              }

              if (!crossed_river(c, s))
                  break;
          }

      // Generate attacks for advisors
      for (Square s : {SQ_D0, SQ_F0, SQ_D2, SQ_F2})
          PseudoAttacks[ADVISOR][s] |= SQ_E1;
      for (Square s : {SQ_D7, SQ_F7, SQ_D9, SQ_F9})
          PseudoAttacks[ADVISOR][s] |= SQ_E8;
      PseudoAttacks[ADVISOR][SQ_E1] |= SquareBB[SQ_D0] | SQ_F0 | SQ_D2 | SQ_F2;
      PseudoAttacks[ADVISOR][SQ_E8] |= SquareBB[SQ_D7] | SQ_F7 | SQ_D9 | SQ_F9;

      // Generate attacks for generals
      for (Square s = SQ_D0; s <= SQ_F9; ++s) {
          if (in_palace(s))
              for (Direction d : {SOUTH, WEST, EAST, NORTH}) {
                  Square to = s + d;
                  if (is_ok(to) && in_palace(to))
                      PseudoAttacks[GENERAL][s] |= to;
              }
      }
  }

  init_magics(CANNON, CannonTable, CannonMagics);
  init_magics(CHARIOT, ChariotTable, ChariotMagics);
  init_magics(HORSE, HorseTable, HorseMagics);
  init_magics(ELEPHANT, ElephantTable, ElephantMagics);

  for (Square s1 = SQ_A0; s1 <= SQ_I9; ++s1)
  {
      PseudoAttacks[  CANNON][s1] = attacks_bb<  CANNON>(s1, 0);
      PseudoAttacks[ CHARIOT][s1] = attacks_bb< CHARIOT>(s1, 0);
      PseudoAttacks[   HORSE][s1] = attacks_bb<   HORSE>(s1, 0);
      PseudoAttacks[ELEPHANT][s1] = attacks_bb<ELEPHANT>(s1, 0);

      for (Square s2 = SQ_A0; s2 <= SQ_I9; ++s2)
      {
          if (!(PseudoAttacks[CHARIOT][s1] & s2))
              continue;

          LineBB[s1][s2] = (attacks_bb(CHARIOT, s1, 0) & attacks_bb(CHARIOT, s2, 0)) | s1 | s2;
          BetweenBB[s1][s2] = attacks_bb(CHARIOT, s1, SquareBB[s2]) & attacks_bb(CHARIOT, s2, SquareBB[s1]);
      }
  }
}


namespace {

  Bitboard sliding_attack(PieceType pt, Square sq, Bitboard occupied) {

    assert(pt == CANNON || pt == CHARIOT);

    Bitboard attack = 0;

    for (Direction d : {SOUTH, WEST, EAST, NORTH})
    {
        bool jumped = false;
        for (Square s = sq + d;
             is_ok(s) && distance(s, s - d) == 1;
             s += d)
        {
            if (pt == CANNON)
            {
                if (jumped)
                {
                    if (occupied & s)
                    {
                        attack |= s;
                        break;
                    }
                }
                else if (occupied & s)
                    jumped = true;
                else
                    attack |= s;
            }
            else
            {
                attack |= s;

                if (occupied & s)
                    break;
            }
        }
    }

    return attack;
  }

  Bitboard horse_attack(Square sq, Bitboard occupied, Bitboard mask) {

    Bitboard attack = 0;

    while (mask)
    {
        Square sqmask = pop_lsb(&mask);
        if (~occupied & sqmask)
        {
            Direction d = Direction(sqmask - sq);
            if (file_of(sqmask) != FILE_A && (d == SOUTH || d == NORTH))
                attack |= sqmask + d + WEST;
            if (file_of(sqmask) != FILE_I && (d == SOUTH || d == NORTH))
                attack |= sqmask + d + EAST;
            if (rank_of(sqmask) != RANK_0 && (d == WEST || d == EAST))
                attack |= sqmask + d + SOUTH;
            if (rank_of(sqmask) != RANK_9 && (d == WEST || d == EAST))
                attack |= sqmask + d + NORTH;
        }
    }
    return attack;
  }

  Bitboard elephant_attack(Square sq, Bitboard occupied, Bitboard mask) {

    Bitboard attack = 0;

    while (mask)
    {
        Square sqmask = pop_lsb(&mask);
        if (~occupied & sqmask)
            attack |= sqmask + Direction(sqmask - sq);
    }
    return attack;
  }

  // init_magics() computes all rook and bishop attacks at startup. Magic
  // bitboards are used to look up attacks of sliding pieces. As a reference see
  // chessprogramming.wikispaces.com/Magic+Bitboards. In particular, here we
  // use the so called "fancy" approach.

  void init_magics(PieceType pt, Bitboard table[], Magic magics[]) {

    assert(pt == CANNON || pt == CHARIOT || pt == HORSE || pt == ELEPHANT);

    // Optimal PRNG seeds to pick the correct magics in the shortest time
    /* int seeds[][RANK_NB] = { { 8977, 44560, 54343, 38998,  5731, 95205, 104912, 17020 },
                             {  728, 10316, 55013, 32803, 12281, 15100,  16645,   255 } }; */

    Bitboard /* occupancy[131072], */* reference = new Bitboard[131072], edges, b;
    int /* epoch[4096] = {}, cnt = 0, */size = 0;

    for (Square s = SQ_A0; s <= SQ_I9; ++s)
    {
        // Board edges are not considered in the relevant occupancies
        edges = (((Bitboard)Rank0BB | Rank9BB) & ~rank_bb(s)) | (((Bitboard)FileABB | FileIBB) & ~file_bb(s));

        // Given a square 's', the mask is the bitboard of sliding attacks from
        // 's' computed on an empty board. The index must be big enough to contain
        // all the attacks for each possible subset of the mask and so is 2 power
        // the number of 1s of the mask. Hence we deduce the size of the shift to
        // apply to the 64 or 32 bits word to get the index.
        Magic& m = magics[s];
        switch (pt) {
        case CANNON:
            m.mask = sliding_attack(CANNON, s, 0);
            break;
        case CHARIOT:
            m.mask = sliding_attack(CHARIOT, s, 0) & ~edges;
            break;
        case HORSE:
            m.mask = 0;
            for (Direction d : {SOUTH, WEST, EAST, NORTH})
            {
                Square mask = s + d;
                if (is_ok(mask) && distance(mask, s) == 1)
                    m.mask |= mask;
            }
            m.mask &= ~edges;
            break;
        default:
            m.mask = 0;
            switch (s) {
            case SQ_C0: case SQ_G0: case SQ_A2: case SQ_E2: case SQ_I2: case SQ_C4: case SQ_G4:
            case SQ_C5: case SQ_G5: case SQ_A7: case SQ_E7: case SQ_I7: case SQ_C9: case SQ_G9:
                for (Direction d : {SOUTH_WEST, SOUTH_EAST, NORTH_WEST, NORTH_EAST})
                {
                    Square mask = s + d;
                    if (is_ok(mask) && distance(mask, s) == 2 && crossed_river(RED, s) == crossed_river(RED, mask))
                        m.mask |= mask;
                }
                break;
            default:
                break;
            }
            break;
        }
        // m.shift = (Is64Bit ? 64 : 32) - popcount(m.mask);

        // Set the offset for the attacks table of the square. We have individual
        // table sizes for each square with "Fancy Magic Bitboards".
        m.attacks = s == SQ_A0 ? table : magics[s - 1].attacks + size;

        // Use Carry-Rippler trick to enumerate all subsets of masks[s] and
        // store the corresponding sliding attack bitboard in reference[].
        b = size = 0;
        do {
            // occupancy[size] = b;
            switch (pt) {
            case CANNON:
                reference[size] = sliding_attack(CANNON, s, b);
                break;
            case CHARIOT:
                reference[size] = sliding_attack(CHARIOT, s, b);
                break;
            case HORSE:
                reference[size] = horse_attack(s, b, m.mask);
                break;
            default:
                reference[size] = elephant_attack(s, b, m.mask);
                break;
            }

            // if (HasPext)
                m.attacks[_pext_u128(b, m.mask).value_u64[0]] = reference[size];

            size++;
            b = (b - m.mask) & m.mask;
        } while (b);

        /* if (HasPext)
            continue;

        PRNG rng(seeds[Is64Bit][rank_of(s)]);

        // Find a magic for square 's' picking up an (almost) random number
        // until we find the one that passes the verification test.
        for (int i = 0; i < size; )
        {
            for (m.magic = 0; popcount((m.magic * m.mask) >> 56) < 6; )
                m.magic = rng.sparse_rand<Bitboard>();

            // A good magic must map every possible occupancy to an index that
            // looks up the correct sliding attack in the attacks[s] database.
            // Note that we build up the database for square 's' as a side
            // effect of verifying the magic. Keep track of the attempt count
            // and save it in epoch[], little speed-up trick to avoid resetting
            // m.attacks[] after every failed attempt.
            for (++cnt, i = 0; i < size; ++i)
            {
                unsigned idx = m.index(occupancy[i]);

                if (epoch[idx] < cnt)
                {
                    epoch[idx] = cnt;
                    m.attacks[idx] = reference[i];
                }
                else if (m.attacks[idx] != reference[i])
                    break;
            }
        } */
    }
  }
}
