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

#ifndef BITBOARD_H_INCLUDED
#define BITBOARD_H_INCLUDED

#include <string>
#include <emmintrin.h> // Requires x86 CPU with SSE2 support

#include "types.h"

struct alignas(16) Bitboard {
    Bitboard();
    Bitboard(const uint64_t high, const uint64_t low);
    Bitboard(const int n);
    Bitboard(const Bitboard & source);
    Bitboard(const __m128i & source);

    Bitboard operator-(const Bitboard & source);
    Bitboard operator-(const int n);

    Bitboard & operator=(const Bitboard & source);
    Bitboard operator&(const Bitboard & source);
    Bitboard & operator&=(const Bitboard & source);
    Bitboard operator|(const Bitboard & source);
    Bitboard & operator|=(const Bitboard & source);
    Bitboard operator~() const;
    Bitboard operator^(const Bitboard & source);
    Bitboard & operator^=(const Bitboard & source);
    Bitboard operator<<(const int n) const;
    // Bitboard operator>>(const int n) const;

    operator bool() const;

    union {
        __m128i value;
        uint64_t value_u64[2];
    };
};

namespace Bitboards {

void init();
const std::string pretty(Bitboard b);

}

const Bitboard AllSquares = Bitboard(UINT64_C(0x3FFFFFF), UINT64_C(0xFFFFFFFFFFFFFFFF));

const Bitboard FileABB = Bitboard(UINT64_C(0x20100), UINT64_C(0x8040201008040201));
const Bitboard FileBBB = FileABB << 1;
const Bitboard FileCBB = FileABB << 2;
const Bitboard FileDBB = FileABB << 3;
const Bitboard FileEBB = FileABB << 4;
const Bitboard FileFBB = FileABB << 5;
const Bitboard FileGBB = FileABB << 6;
const Bitboard FileHBB = FileABB << 7;
const Bitboard FileIBB = FileABB << 8;

const Bitboard Rank0BB = Bitboard(UINT64_C(0), UINT64_C(0x1FF));
const Bitboard Rank1BB = Rank0BB << (9 * 1);
const Bitboard Rank2BB = Rank0BB << (9 * 2);
const Bitboard Rank3BB = Rank0BB << (9 * 3);
const Bitboard Rank4BB = Rank0BB << (9 * 4);
const Bitboard Rank5BB = Rank0BB << (9 * 5);
const Bitboard Rank6BB = Rank0BB << (9 * 6);
const Bitboard Rank7BB = Rank0BB << (9 * 7);
const Bitboard Rank8BB = Rank0BB << (9 * 8);
const Bitboard Rank9BB = Rank0BB << (9 * 9);

const Bitboard PalaceBB = (Bitboard(Rank0BB) | Rank1BB | Rank2BB | Rank7BB | Rank8BB | Rank9BB)
                        & (Bitboard(FileDBB) | FileEBB | FileFBB);

extern int SquareDistance[SQUARE_NB][SQUARE_NB];

extern Bitboard SquareBB[SQUARE_NB];
extern Bitboard FileBB[FILE_NB];
extern Bitboard RankBB[RANK_NB];
extern Bitboard AdjacentFilesBB[FILE_NB];
extern Bitboard ForwardRanksBB[COLOR_NB][RANK_NB];
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
// extern Bitboard DistanceRingBB[SQUARE_NB][8];
extern Bitboard ForwardFileBB[COLOR_NB][SQUARE_NB];
// extern Bitboard PassedPawnMask[COLOR_NB][SQUARE_NB];
// extern Bitboard PawnAttackSpan[COLOR_NB][SQUARE_NB];
extern Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
extern Bitboard SoldierAttacks[COLOR_NB][SQUARE_NB];


/// Magic holds all magic bitboards relevant data for a single square
/// Avoid using magics by implementing a slow pext substitute
struct Magic {
  Bitboard  mask;
  // Bitboard  magic;
  Bitboard* attacks;
  // unsigned  shift;

  // Compute the attack's index using the 'magic bitboards' approach
  unsigned index(Bitboard occupied) const {
    Bitboard _pext_u128(Bitboard, Bitboard);
    return unsigned(_pext_u128(occupied, mask).value_u64[0]);
    /* if (HasPext)
        return unsigned(pext(occupied, mask));

    if (Is64Bit)
        return unsigned(((occupied & mask) * magic) >> shift);

    unsigned lo = unsigned(occupied) & unsigned(mask);
    unsigned hi = unsigned(occupied >> 32) & unsigned(mask >> 32);
    return (lo * unsigned(magic) ^ hi * unsigned(magic >> 32)) >> shift; */
  }
};

extern Magic CannonMagics[SQUARE_NB];
extern Magic ChariotMagics[SQUARE_NB];
extern Magic HorseMagics[SQUARE_NB];
extern Magic ElephantMagics[SQUARE_NB];


/// Overloads of bitwise operators between a Bitboard and a Square for testing
/// whether a given bit is set in a bitboard, and for setting and clearing bits.

inline Bitboard operator&(Bitboard b, Square s) {
  return b & SquareBB[s];
}

inline Bitboard operator|(Bitboard b, Square s) {
  return b | SquareBB[s];
}

inline Bitboard operator^(Bitboard b, Square s) {
  return b ^ SquareBB[s];
}

inline Bitboard& operator|=(Bitboard& b, Square s) {
  return b |= SquareBB[s];
}

inline Bitboard& operator^=(Bitboard& b, Square s) {
  return b ^= SquareBB[s];
}

inline const bool more_than_one(Bitboard b) {
  return b & (b - 1);
}

/// rank_bb() and file_bb() return a bitboard representing all the squares on
/// the given file or rank.

inline Bitboard rank_bb(Rank r) {
  return RankBB[r];
}

inline Bitboard rank_bb(Square s) {
  return RankBB[rank_of(s)];
}

inline Bitboard file_bb(File f) {
  return FileBB[f];
}

inline Bitboard file_bb(Square s) {
  return FileBB[file_of(s)];
}


/// make_bitboard() returns a bitboard from a list of squares

inline const Bitboard make_bitboard() { return 0; }

template<typename ...Squares>
constexpr Bitboard make_bitboard(Square s, Squares... squares) {
  return (Bitboard(UINT64_C(0), UINT64_C(1)) << s) | make_bitboard(squares...);
}


/// shift() moves a bitboard one step along direction D (mainly for pawns)

template<Direction D>
constexpr Bitboard shift(Bitboard b) {
  return  D == NORTH      ?  b             <<  9 : D == SOUTH      ?  b             >>  9
        : D == EAST       ? (b & ~FileHBB) <<  1 : D == WEST       ? (b & ~FileABB) >>  1
        : D == NORTH_EAST ? (b & ~FileHBB) << 10 : D == NORTH_WEST ? (b & ~FileABB) <<  8
        : D == SOUTH_EAST ? (b & ~FileHBB) >>  8 : D == SOUTH_WEST ? (b & ~FileABB) >> 10
        : 0;
}


/// pawn_attacks_bb() returns the pawn attacks for the given color from the
/// squares in the given bitboard.

/* template<Color C>
constexpr Bitboard pawn_attacks_bb(Bitboard b) {
  return C == WHITE ? shift<NORTH_WEST>(b) | shift<NORTH_EAST>(b)
                    : shift<SOUTH_WEST>(b) | shift<SOUTH_EAST>(b);
} */


/// adjacent_files_bb() returns a bitboard representing all the squares on the
/// adjacent files of the given one.

inline Bitboard adjacent_files_bb(File f) {
  return AdjacentFilesBB[f];
}


/// in_palace() returns true if a square is in palace.

inline bool in_palace(Square s) {
  return PalaceBB & s;
}


/// between_bb() returns a bitboard representing all the squares between the two
/// given ones. For instance, between_bb(SQ_C4, SQ_F7) returns a bitboard with
/// the bits for square d5 and e6 set. If s1 and s2 are not on the same rank, file
/// or diagonal, 0 is returned.

inline Bitboard between_bb(Square s1, Square s2) {
  return BetweenBB[s1][s2];
}


/// forward_ranks_bb() returns a bitboard representing the squares on all the ranks
/// in front of the given one, from the point of view of the given color. For instance,
/// forward_ranks_bb(BLACK, SQ_D3) will return the 16 squares on ranks 1 and 2.

inline Bitboard forward_ranks_bb(Color c, Square s) {
  return ForwardRanksBB[c][rank_of(s)];
}


/// forward_file_bb() returns a bitboard representing all the squares along the line
/// in front of the given one, from the point of view of the given color:
///      ForwardFileBB[c][s] = forward_ranks_bb(c, s) & file_bb(s)

inline Bitboard forward_file_bb(Color c, Square s) {
  return ForwardFileBB[c][s];
}


/// pawn_attack_span() returns a bitboard representing all the squares that can be
/// attacked by a pawn of the given color when it moves along its file, starting
/// from the given square:
///      PawnAttackSpan[c][s] = forward_ranks_bb(c, s) & adjacent_files_bb(file_of(s));

/* inline Bitboard pawn_attack_span(Color c, Square s) {
  return PawnAttackSpan[c][s];
} */


/// passed_pawn_mask() returns a bitboard mask which can be used to test if a
/// pawn of the given color and on the given square is a passed pawn:
///      PassedPawnMask[c][s] = pawn_attack_span(c, s) | forward_file_bb(c, s)

/* inline Bitboard passed_pawn_mask(Color c, Square s) {
  return PassedPawnMask[c][s];
} */


/// aligned() returns true if the squares s1, s2 and s3 are aligned either on a
/// straight or on a diagonal line.

inline bool aligned(Square s1, Square s2, Square s3) {
  return LineBB[s1][s2] & s3;
}


/// distance() functions return the distance between x and y, defined as the
/// number of steps for a general in x to reach y. Works with squares, ranks, files.

template<typename T> inline int distance(T x, T y) { return x < y ? y - x : x - y; }
template<> inline int distance<Square>(Square x, Square y) { return SquareDistance[x][y]; }

template<typename T1, typename T2> inline int distance(T2 x, T2 y);
template<> inline int distance<File>(Square x, Square y) { return distance(file_of(x), file_of(y)); }
template<> inline int distance<Rank>(Square x, Square y) { return distance(rank_of(x), rank_of(y)); }


/// attacks_bb() returns a bitboard representing all the squares attacked by a
/// piece of type Pt (bishop or rook) placed on 's'.

template<PieceType Pt>
inline Bitboard attacks_bb(Square s, Bitboard occupied) {

  assert(Pt == CANNON || Pt == CHARIOT || Pt == HORSE || Pt == ELEPHANT);
  const Magic& m = (Pt ==  CANNON) ? CannonMagics[s]
                 : (Pt == CHARIOT) ? ChariotMagics[s]
                 : (Pt ==   HORSE) ? HorseMagics[s]
                 : ElephantMagics[s];
  return m.attacks[m.index(occupied)];
}

inline Bitboard attacks_bb(PieceType pt, Square s, Bitboard occupied) {

  switch (pt)
  {
  case CANNON  : return attacks_bb<  CANNON>(s, occupied);
  case CHARIOT : return attacks_bb< CHARIOT>(s, occupied);
  case HORSE   : return attacks_bb<   HORSE>(s, occupied);
  case ELEPHANT: return attacks_bb<ELEPHANT>(s, occupied);
  default      : return PseudoAttacks[pt][s];
  }
}


/// popcount() counts the number of non-zero bits in a bitboard

inline int popcount(uint64_t b) {

#ifndef USE_POPCNT

  extern uint8_t PopCnt16[1 << 16];
  union { uint64_t bb; uint16_t u[4]; } v = { b };
  return PopCnt16[v.u[0]] + PopCnt16[v.u[1]] + PopCnt16[v.u[2]] + PopCnt16[v.u[3]];

#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)

  return (int)_mm_popcnt_u64(b);

#else // Assumed gcc or compatible compiler

  return __builtin_popcountll(b);

#endif
}

inline int popcount(Bitboard b) {

#ifndef USE_POPCNT

  extern uint8_t PopCnt16[1 << 16];
  union { Bitboard bb; uint16_t u[8]; } v = { b };
  return PopCnt16[v.u[0]] + PopCnt16[v.u[1]] + PopCnt16[v.u[2]] + PopCnt16[v.u[3]] + PopCnt16[v.u[4]] + PopCnt16[v.u[5]] + PopCnt16[v.u[6]] + PopCnt16[v.u[7]];

#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)

  return (int)(_mm_popcnt_u64(b.value_u64[0]) + _mm_popcnt_u64(b.value_u64[1]));

#else // Assumed gcc or compatible compiler

  return __builtin_popcountll(b.value_u64[0]) + __builtin_popcountll(b.value_u64[1]);

#endif
}

inline uint64_t __pext_u64(uint64_t a, uint64_t mask) {
    uint64_t ret = 0;
    for (uint64_t i = 1; mask; i <<= 1) {
        if (a & mask & (uint64_t)-(int64_t)mask)
            ret |= i;
        mask &= mask - 1;
    }
    return ret;
}

inline Bitboard _pext_u128(Bitboard a, Bitboard mask)
{
    return (Bitboard(UINT64_C(0), pext(a.value_u64[1], mask.value_u64[1]))
        << (popcount(mask.value_u64[0])))
        | Bitboard(UINT64_C(0), pext(a.value_u64[0], mask.value_u64[0]));
}

/// lsb() and msb() return the least/most significant bit in a non-zero bitboard

#if defined(__GNUC__)  // GCC, Clang, ICC

inline Square lsb(Bitboard b) {
  assert(b);

  if (b.value_u64[0])
      return Square(__builtin_ctzll(b.value_u64[0]));
  else
      return Square(__builtin_ctzll(b.value_u64[1]) + 64);
}

inline Square msb(Bitboard b) {
  assert(b);

  if (b.value_u64[1])
      return Square(127 ^ __builtin_clzll(b.value_u64[1]));
  else
      return Square(63 ^ __builtin_clzll(b.value_u64[0]));
}

#elif defined(_MSC_VER)  // MSVC

#ifdef _WIN64  // MSVC, WIN64

inline Square lsb(Bitboard b) {
  assert(b);
  unsigned long idx;

  if (b.value_u64[0]) {
      _BitScanForward64(&idx, b.value_u64[0]);
      return Square(idx);
  } else {
      _BitScanForward64(&idx, b.value_u64[1]);
      return Square(idx + 64UL);
  }
}

inline Square msb(Bitboard b) {
  assert(b);
  unsigned long idx;

  if (b.value_u64[1]) {
      _BitScanReverse64(&idx, b.value_u64[1]);
      return Square(idx + 64UL);
  } else {
      _BitScanReverse64(&idx, b.value_u64[0]);
      return Square(idx);
  }
}

#else  // MSVC, WIN32

inline Square lsb(Bitboard b) {
  assert(b);
  unsigned long idx;

  if (b.value_u64[0]) {
    if (b.value_u64[0] & 0xffffffff) {
      _BitScanForward(&idx, int32_t(b.value_u64[0]));
      return Square(idx);
    } else {
      _BitScanForward(&idx, int32_t(b.value_u64[0] >> 32));
      return Square(idx + 32);
    }
  } else {
    if (b.value_u64[1] & 0xffffffff) {
      _BitScanForward(&idx, int32_t(b.value_u64[1]));
      return Square(idx + 64);
    } else {
      _BitScanForward(&idx, int32_t(b.value_u64[1] >> 32));
      return Square(idx + 96);
    }
  }
}

inline Square msb(Bitboard b) {
  assert(b);
  unsigned long idx;

  if (b.value_u64[1]) {
    if (b.value_u64[1] >> 32) {
      _BitScanReverse(&idx, int32_t(b.value_u64[1] >> 32));
      return Square(idx + 96);
    } else {
      _BitScanReverse(&idx, int32_t(b.value_u64[1]));
      return Square(idx + 64);
    }
  } else {
    if (b.value_u64[0] >> 32) {
      _BitScanReverse(&idx, int32_t(b.value_u64[0] >> 32));
      return Square(idx + 32);
    } else {
      _BitScanReverse(&idx, int32_t(b.value_u64[0]));
      return Square(idx);
    }
  }
}

#endif

#else  // Compiler is neither GCC nor MSVC compatible

#error "Compiler not supported."

#endif


/// pop_lsb() finds and clears the least significant bit in a non-zero bitboard

inline Square pop_lsb(Bitboard* b) {
  const Square s = lsb(*b);
  *b &= *b - 1;
  return s;
}


/// frontmost_sq() and backmost_sq() return the square corresponding to the
/// most/least advanced bit relative to the given color.

inline Square frontmost_sq(Color c, Bitboard b) { return c == RED ? msb(b) : lsb(b); }
inline Square  backmost_sq(Color c, Bitboard b) { return c == RED ? lsb(b) : msb(b); }

#endif // #ifndef BITBOARD_H_INCLUDED
