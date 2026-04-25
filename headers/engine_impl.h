#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#ifndef _WIN32
#include <cwchar>
#endif

#include "engine.h"

#ifndef FASTAWC_BACKEND_NAMESPACE
#error FASTAWC_BACKEND_NAMESPACE must be defined before including engine_impl.h
#endif

#ifndef FASTAWC_BACKEND_NAME
#error FASTAWC_BACKEND_NAME must be defined before including engine_impl.h
#endif

#ifndef FASTAWC_BACKEND_ENABLE_AVX2
#error FASTAWC_BACKEND_ENABLE_AVX2 must be defined before including engine_impl.h
#endif

namespace fastawc {
namespace FASTAWC_BACKEND_NAMESPACE {

FASTAWC_FORCEINLINE bool is_space_ascii(const uint8_t c) noexcept {
	return c == ' ' || static_cast<unsigned>(c - '\t') < 5u;
}

FASTAWC_FORCEINLINE bool is_utf8_continuation(const uint8_t c) noexcept {
	return (c & 0xC0u) == 0x80u;
}

FASTAWC_FORCEINLINE uint32_t is_utf8_lead(const uint8_t c) noexcept {
	return static_cast<uint32_t>((c & 0xC0u) != 0x80u);
}

FASTAWC_FORCEINLINE uint32_t low_mask32(const unsigned bits) noexcept {
	return bits >= 32 ? 0xFFFFFFFFu : static_cast<uint32_t>((1ull << bits) - 1ull);
}

FASTAWC_FORCEINLINE uint32_t range_mask32(const unsigned begin, const unsigned end) noexcept {
	return low_mask32(end) & ~low_mask32(begin);
}

FASTAWC_FORCEINLINE void prefetch_read(const void* const ptr) noexcept {
#if FASTAWC_BACKEND_ENABLE_AVX2
	_mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
	__builtin_prefetch(ptr, 0, 3);
#else
	(void)ptr;
#endif
}

FASTAWC_FORCEINLINE bool is_unicode_whitespace(const uint32_t codePoint) noexcept {
	if (codePoint <= 0x7Fu) {
		return codePoint == ' ' || static_cast<unsigned>(codePoint - '\t') < 5u;
	}

	switch (codePoint) {
	case 0x0085u:
	case 0x00A0u:
	case 0x1680u:
	case 0x2028u:
	case 0x2029u:
	case 0x202Fu:
	case 0x205Fu:
	case 0x2060u:
	case 0x3000u:
		return true;
	default:
		break;
	}

	return codePoint >= 0x2000u && codePoint <= 0x200Au;
}

FASTAWC_FORCEINLINE uint64_t tab_advance(const uint64_t currentWidth) noexcept {
	return 8u - (currentWidth & 7u);
}

#ifdef _WIN32
struct UnicodeRange {
	uint32_t begin;
	uint32_t end;
};

template<size_t N>
FASTAWC_FORCEINLINE bool codepoint_in_ranges(
	const uint32_t codePoint,
	const UnicodeRange(&ranges)[N]) noexcept
{
	size_t lo = 0;
	size_t hi = N;
	while (lo < hi) {
		const size_t mid = lo + ((hi - lo) >> 1);
		const UnicodeRange range = ranges[mid];
		if (codePoint < range.begin) {
			hi = mid;
		}
		else if (codePoint > range.end) {
			lo = mid + 1;
		}
		else {
			return true;
		}
	}
	return false;
}

FASTAWC_FORCEINLINE bool is_zero_width_codepoint(const uint32_t codePoint) noexcept {
	static constexpr UnicodeRange kZeroWidthRanges[] = {
		{0x0300u, 0x036Fu}, {0x0483u, 0x0489u}, {0x0591u, 0x05BDu}, {0x05BFu, 0x05BFu},
		{0x05C1u, 0x05C2u}, {0x05C4u, 0x05C5u}, {0x05C7u, 0x05C7u}, {0x0610u, 0x061Au},
		{0x064Bu, 0x065Fu}, {0x0670u, 0x0670u}, {0x06D6u, 0x06DDu}, {0x06DFu, 0x06E4u},
		{0x06E7u, 0x06E8u}, {0x06EAu, 0x06EDu}, {0x0711u, 0x0711u}, {0x0730u, 0x074Au},
		{0x07A6u, 0x07B0u}, {0x07EBu, 0x07F3u}, {0x07FDu, 0x07FDu}, {0x0816u, 0x0819u},
		{0x081Bu, 0x0823u}, {0x0825u, 0x0827u}, {0x0829u, 0x082Du}, {0x0859u, 0x085Bu},
		{0x0898u, 0x089Fu}, {0x08CAu, 0x08E1u}, {0x08E3u, 0x0902u}, {0x093Au, 0x093Cu},
		{0x093Eu, 0x094Du}, {0x0951u, 0x0957u},
		{0x0962u, 0x0963u}, {0x0981u, 0x0981u}, {0x09BCu, 0x09BCu}, {0x09C1u, 0x09C4u},
		{0x09CDu, 0x09CDu}, {0x09E2u, 0x09E3u}, {0x09FEu, 0x09FEu}, {0x0A01u, 0x0A02u},
		{0x0A3Cu, 0x0A3Cu}, {0x0A41u, 0x0A42u}, {0x0A47u, 0x0A48u}, {0x0A4Bu, 0x0A4Du},
		{0x0A51u, 0x0A51u}, {0x0A70u, 0x0A71u}, {0x0A75u, 0x0A75u}, {0x0A81u, 0x0A82u},
		{0x0ABCu, 0x0ABCu}, {0x0AC1u, 0x0AC5u}, {0x0AC7u, 0x0AC8u}, {0x0ACDu, 0x0ACDu},
		{0x0AE2u, 0x0AE3u}, {0x0AFAu, 0x0AFFu}, {0x0B01u, 0x0B01u}, {0x0B3Cu, 0x0B3Cu},
		{0x0B3Fu, 0x0B3Fu}, {0x0B41u, 0x0B44u}, {0x0B4Du, 0x0B4Du}, {0x0B55u, 0x0B56u},
		{0x0B62u, 0x0B63u}, {0x0B82u, 0x0B82u}, {0x0BC0u, 0x0BC0u}, {0x0BCDu, 0x0BCDu},
		{0x0C00u, 0x0C00u}, {0x0C04u, 0x0C04u}, {0x0C3Eu, 0x0C40u}, {0x0C46u, 0x0C48u},
		{0x0C4Au, 0x0C4Du}, {0x0C55u, 0x0C56u}, {0x0C62u, 0x0C63u}, {0x0C81u, 0x0C81u},
		{0x0CBCu, 0x0CBCu}, {0x0CBFu, 0x0CBFu}, {0x0CC6u, 0x0CC6u}, {0x0CCCu, 0x0CCDu},
		{0x0CE2u, 0x0CE3u}, {0x0D00u, 0x0D01u}, {0x0D3Bu, 0x0D3Cu}, {0x0D41u, 0x0D44u},
		{0x0D4Du, 0x0D4Du}, {0x0D62u, 0x0D63u}, {0x0D81u, 0x0D81u}, {0x0DCAu, 0x0DCAu},
		{0x0DD2u, 0x0DD4u}, {0x0DD6u, 0x0DD6u}, {0x0E31u, 0x0E31u}, {0x0E34u, 0x0E3Au},
		{0x0E47u, 0x0E4Eu}, {0x0EB1u, 0x0EB1u}, {0x0EB4u, 0x0EBCu}, {0x0EC8u, 0x0ECDu},
		{0x0F18u, 0x0F19u}, {0x0F35u, 0x0F35u}, {0x0F37u, 0x0F37u}, {0x0F39u, 0x0F39u},
		{0x0F71u, 0x0F7Eu}, {0x0F80u, 0x0F84u}, {0x0F86u, 0x0F87u}, {0x0F8Du, 0x0F97u},
		{0x0F99u, 0x0FBCu}, {0x0FC6u, 0x0FC6u}, {0x102Du, 0x1030u}, {0x1032u, 0x1037u},
		{0x1039u, 0x103Au}, {0x103Du, 0x103Eu}, {0x1058u, 0x1059u}, {0x105Eu, 0x1060u},
		{0x1071u, 0x1074u}, {0x1082u, 0x1082u}, {0x1085u, 0x1086u}, {0x108Du, 0x108Du},
		{0x109Du, 0x109Du}, {0x135Du, 0x135Fu}, {0x1712u, 0x1714u}, {0x1732u, 0x1734u},
		{0x1752u, 0x1753u}, {0x1772u, 0x1773u}, {0x17B4u, 0x17B5u}, {0x17B7u, 0x17BDu},
		{0x17C6u, 0x17C6u}, {0x17C9u, 0x17D3u}, {0x17DDu, 0x17DDu}, {0x180Bu, 0x180Fu},
		{0x1885u, 0x1886u}, {0x18A9u, 0x18A9u}, {0x1920u, 0x1922u}, {0x1927u, 0x1928u},
		{0x1932u, 0x1932u}, {0x1939u, 0x193Bu}, {0x1A17u, 0x1A18u}, {0x1A1Bu, 0x1A1Bu},
		{0x1A56u, 0x1A56u}, {0x1A58u, 0x1A5Eu}, {0x1A60u, 0x1A60u}, {0x1A62u, 0x1A62u},
		{0x1A65u, 0x1A6Cu}, {0x1A73u, 0x1A7Cu}, {0x1A7Fu, 0x1A7Fu}, {0x1AB0u, 0x1ACEu},
		{0x1B00u, 0x1B03u}, {0x1B34u, 0x1B34u}, {0x1B36u, 0x1B3Au}, {0x1B3Cu, 0x1B3Cu},
		{0x1B42u, 0x1B42u}, {0x1B6Bu, 0x1B73u}, {0x1B80u, 0x1B81u}, {0x1BA2u, 0x1BA5u},
		{0x1BA8u, 0x1BA9u}, {0x1BABu, 0x1BADu}, {0x1BE6u, 0x1BE6u}, {0x1BE8u, 0x1BE9u},
		{0x1BEDu, 0x1BEDu}, {0x1BEFu, 0x1BF1u}, {0x1C2Cu, 0x1C33u}, {0x1C36u, 0x1C37u},
		{0x1CD0u, 0x1CD2u}, {0x1CD4u, 0x1CE0u}, {0x1CE2u, 0x1CE8u}, {0x1CEDu, 0x1CEDu},
		{0x1CF4u, 0x1CF4u}, {0x1CF8u, 0x1CF9u}, {0x1DC0u, 0x1DFFu}, {0x200Bu, 0x200Fu},
		{0x202Au, 0x202Eu}, {0x2060u, 0x2064u}, {0x2066u, 0x206Fu}, {0x20D0u, 0x20F0u},
		{0x2CEF0u, 0x2CEF0u}
	};
	static constexpr UnicodeRange kZeroWidthTailRanges[] = {
		{0x2D7Fu, 0x2D7Fu}, {0x2DE0u, 0x2DFFu}, {0x302Au, 0x302Fu}, {0x3099u, 0x309Au},
		{0xA66Fu, 0xA672u}, {0xA674u, 0xA67Du}, {0xA69Eu, 0xA69Fu}, {0xA6F0u, 0xA6F1u},
		{0xA802u, 0xA802u}, {0xA806u, 0xA806u}, {0xA80Bu, 0xA80Bu}, {0xA825u, 0xA826u},
		{0xA8C4u, 0xA8C5u}, {0xA8E0u, 0xA8F1u}, {0xA8FFu, 0xA8FFu}, {0xA926u, 0xA92Du},
		{0xA947u, 0xA951u}, {0xA980u, 0xA982u}, {0xA9B3u, 0xA9B3u}, {0xA9B6u, 0xA9B9u},
		{0xA9BCu, 0xA9BDu}, {0xA9E5u, 0xA9E5u}, {0xAA29u, 0xAA2Eu}, {0xAA31u, 0xAA32u},
		{0xAA35u, 0xAA36u}, {0xAA43u, 0xAA43u}, {0xAA4Cu, 0xAA4Cu}, {0xAA7Cu, 0xAA7Cu},
		{0xAAB0u, 0xAAB0u}, {0xAAB2u, 0xAAB4u}, {0xAAB7u, 0xAAB8u}, {0xAABEu, 0xAABFu},
		{0xAAC1u, 0xAAC1u}, {0xAAECu, 0xAAEDu}, {0xAAF6u, 0xAAF6u}, {0xABE5u, 0xABE5u},
		{0xABE8u, 0xABE8u}, {0xABEDu, 0xABEDu}, {0xFB1Eu, 0xFB1Eu}, {0xFE00u, 0xFE0Fu},
		{0xFE20u, 0xFE2Fu}, {0xFEFFu, 0xFEFFu}, {0xFFF9u, 0xFFFBu}, {0x101FDu, 0x101FDu},
		{0x102E0u, 0x102E0u}, {0x10376u, 0x1037Au}, {0x10A01u, 0x10A03u}, {0x10A05u, 0x10A06u},
		{0x10A0Cu, 0x10A0Fu}, {0x10A38u, 0x10A3Au}, {0x10A3Fu, 0x10A3Fu}, {0x10AE5u, 0x10AE6u},
		{0x10D24u, 0x10D27u}, {0x10EABu, 0x10EAFu}, {0x10F46u, 0x10F50u}, {0x10F82u, 0x10F85u},
		{0x11001u, 0x11001u}, {0x11038u, 0x11046u}, {0x11070u, 0x11070u}, {0x11073u, 0x11074u},
		{0x1107Fu, 0x11081u}, {0x110B3u, 0x110B6u}, {0x110B9u, 0x110BAu}, {0x11100u, 0x11102u},
		{0x11127u, 0x1112Bu}, {0x1112Du, 0x11134u}, {0x11173u, 0x11173u}, {0x11180u, 0x11181u},
		{0x111B6u, 0x111BEu}, {0x111C9u, 0x111CCu}, {0x111CFu, 0x111CFu}, {0x1122Fu, 0x11231u},
		{0x11234u, 0x11234u}, {0x11236u, 0x11237u}, {0x1123Eu, 0x1123Eu}, {0x112DFu, 0x112DFu},
		{0x112E3u, 0x112EAu}, {0x11300u, 0x11301u}, {0x1133Bu, 0x1133Cu}, {0x11340u, 0x11340u},
		{0x11366u, 0x1136Cu}, {0x11370u, 0x11374u}, {0x11438u, 0x1143Fu}, {0x11442u, 0x11444u},
		{0x11446u, 0x11446u}, {0x1145Eu, 0x1145Eu}, {0x114B3u, 0x114B8u}, {0x114BAu, 0x114BAu},
		{0x114BFu, 0x114C0u}, {0x114C2u, 0x114C3u}, {0x115B2u, 0x115B5u}, {0x115BCu, 0x115BDu},
		{0x115BFu, 0x115C0u}, {0x115DCu, 0x115DDu}, {0x11633u, 0x1163Au}, {0x1163Du, 0x1163Du},
		{0x1163Fu, 0x11640u}, {0x116ABu, 0x116ABu}, {0x116ADu, 0x116ADu}, {0x116B0u, 0x116B5u},
		{0x116B7u, 0x116B7u}, {0x1171Du, 0x1171Fu}, {0x11722u, 0x11725u}, {0x11727u, 0x1172Bu},
		{0x1182Fu, 0x11837u}, {0x11839u, 0x1183Au}, {0x1193Bu, 0x1193Cu}, {0x1193Eu, 0x1193Eu},
		{0x11943u, 0x11943u}, {0x119D4u, 0x119D7u}, {0x119DAu, 0x119DBu}, {0x119E0u, 0x119E0u},
		{0x11A01u, 0x11A0Au}, {0x11A33u, 0x11A38u}, {0x11A3Bu, 0x11A3Eu}, {0x11A47u, 0x11A47u},
		{0x11A51u, 0x11A56u}, {0x11A59u, 0x11A5Bu}, {0x11A8Au, 0x11A96u}, {0x11A98u, 0x11A99u},
		{0x11C30u, 0x11C36u}, {0x11C38u, 0x11C3Du}, {0x11C3Fu, 0x11C3Fu}, {0x11C92u, 0x11CA7u},
		{0x11CAAu, 0x11CB0u}, {0x11CB2u, 0x11CB3u}, {0x11CB5u, 0x11CB6u}, {0x11D31u, 0x11D36u},
		{0x11D3Au, 0x11D3Au}, {0x11D3Cu, 0x11D3Du}, {0x11D3Fu, 0x11D45u}, {0x11D47u, 0x11D47u},
		{0x11D90u, 0x11D91u}, {0x11D95u, 0x11D95u}, {0x11D97u, 0x11D97u}, {0x11EF3u, 0x11EF4u},
		{0x13430u, 0x13438u}, {0x16AF0u, 0x16AF4u}, {0x16B30u, 0x16B36u}, {0x16F4Fu, 0x16F4Fu},
		{0x16F8Fu, 0x16F92u}, {0x16FE4u, 0x16FE4u}, {0x1BC9Du, 0x1BC9Eu}, {0x1CF00u, 0x1CF2Du},
		{0x1CF30u, 0x1CF46u}, {0x1D167u, 0x1D169u}, {0x1D17Bu, 0x1D182u}, {0x1D185u, 0x1D18Bu},
		{0x1D1AAu, 0x1D1ADu}, {0x1D242u, 0x1D244u}, {0x1DA00u, 0x1DA36u}, {0x1DA3Bu, 0x1DA6Cu},
		{0x1DA75u, 0x1DA75u}, {0x1DA84u, 0x1DA84u}, {0x1DA9Bu, 0x1DA9Fu}, {0x1DAA1u, 0x1DAAFu},
		{0x1E000u, 0x1E006u}, {0x1E008u, 0x1E018u}, {0x1E01Bu, 0x1E021u}, {0x1E023u, 0x1E024u},
		{0x1E026u, 0x1E02Au}, {0x1E08Fu, 0x1E08Fu}, {0x1E130u, 0x1E136u}, {0x1E2AEu, 0x1E2AEu},
		{0x1E2ECu, 0x1E2EFu}, {0x1E4ECu, 0x1E4EFu}, {0x1E8D0u, 0x1E8D6u}, {0x1E944u, 0x1E94Au},
		{0xE0001u, 0xE0001u}, {0xE0020u, 0xE007Fu}, {0xE0100u, 0xE01EFu}
	};
	return codepoint_in_ranges(codePoint, kZeroWidthRanges) || codepoint_in_ranges(codePoint, kZeroWidthTailRanges);
}

FASTAWC_FORCEINLINE bool is_wide_codepoint(const uint32_t codePoint) noexcept {
	static constexpr UnicodeRange kWideRanges[] = {
		{0x1100u, 0x115Fu}, {0x231Au, 0x231Bu}, {0x2329u, 0x232Au}, {0x23E9u, 0x23ECu},
		{0x23F0u, 0x23F0u}, {0x23F3u, 0x23F3u}, {0x25FDu, 0x25FEu}, {0x2614u, 0x2615u},
		{0x2648u, 0x2653u}, {0x267Fu, 0x267Fu}, {0x2693u, 0x2693u}, {0x26A1u, 0x26A1u},
		{0x26AAu, 0x26ABu}, {0x26BDu, 0x26BEu}, {0x26C4u, 0x26C5u}, {0x26CEu, 0x26CEu},
		{0x26D4u, 0x26D4u}, {0x26EAu, 0x26EAu}, {0x26F2u, 0x26F3u}, {0x26F5u, 0x26F5u},
		{0x26FAu, 0x26FAu}, {0x26FDu, 0x26FDu}, {0x2705u, 0x2705u}, {0x270Au, 0x270Bu},
		{0x2728u, 0x2728u}, {0x274Cu, 0x274Cu}, {0x274Eu, 0x274Eu}, {0x2753u, 0x2755u},
		{0x2757u, 0x2757u}, {0x2795u, 0x2797u}, {0x27B0u, 0x27B0u}, {0x27BFu, 0x27BFu},
		{0x2B1Bu, 0x2B1Cu}, {0x2B50u, 0x2B50u}, {0x2B55u, 0x2B55u}, {0x2E80u, 0x2FFBu},
		{0x3000u, 0x303Eu}, {0x3041u, 0x33FFu}, {0x3400u, 0x4DBFu}, {0x4E00u, 0xA4C6u},
		{0xA960u, 0xA97Cu}, {0xAC00u, 0xD7A3u}, {0xF900u, 0xFAFFu}, {0xFE10u, 0xFE19u},
		{0xFE30u, 0xFE6Bu}, {0xFF01u, 0xFF60u}, {0xFFE0u, 0xFFE6u}, {0x16FE0u, 0x16FE4u},
		{0x17000u, 0x187F7u}, {0x18800u, 0x18CD5u}, {0x18D00u, 0x18D08u}, {0x1AFF0u, 0x1AFF3u},
		{0x1AFF5u, 0x1AFFBu}, {0x1AFFDu, 0x1AFFEu}, {0x1B000u, 0x1B122u}, {0x1B132u, 0x1B132u},
		{0x1B150u, 0x1B152u}, {0x1B155u, 0x1B155u}, {0x1B164u, 0x1B167u}, {0x1F004u, 0x1F004u},
		{0x1F0CFu, 0x1F0CFu}, {0x1F18Eu, 0x1F18Eu}, {0x1F191u, 0x1F19Au}, {0x1F200u, 0x1F202u},
		{0x1F210u, 0x1F23Bu}, {0x1F240u, 0x1F248u}, {0x1F250u, 0x1F251u}, {0x1F260u, 0x1F265u},
		{0x1F300u, 0x1F320u}, {0x1F32Du, 0x1F335u}, {0x1F337u, 0x1F37Cu}, {0x1F37Eu, 0x1F393u},
		{0x1F3A0u, 0x1F3CAu}, {0x1F3CFu, 0x1F3D3u}, {0x1F3E0u, 0x1F3F0u}, {0x1F3F4u, 0x1F3F4u},
		{0x1F3F8u, 0x1F43Eu}, {0x1F440u, 0x1F440u}, {0x1F442u, 0x1F4FCu}, {0x1F4FFu, 0x1F53Du},
		{0x1F54Bu, 0x1F54Eu}, {0x1F550u, 0x1F567u}, {0x1F57Au, 0x1F57Au}, {0x1F595u, 0x1F596u},
		{0x1F5A4u, 0x1F5A4u}, {0x1F5FBu, 0x1F64Fu}, {0x1F680u, 0x1F6C5u}, {0x1F6CCu, 0x1F6CCu},
		{0x1F6D0u, 0x1F6D2u}, {0x1F6D5u, 0x1F6D7u}, {0x1F6EBu, 0x1F6ECu}, {0x1F6F4u, 0x1F6FCu},
		{0x1F7E0u, 0x1F7EBu}, {0x1F90Cu, 0x1F93Au}, {0x1F93Cu, 0x1F945u}, {0x1F947u, 0x1F9FFu},
		{0x1FA70u, 0x1FA7Cu}, {0x1FA80u, 0x1FA89u}, {0x1FA8Fu, 0x1FAC6u}, {0x1FACEu, 0x1FADCu},
		{0x1FADFu, 0x1FAE9u}, {0x1FAF0u, 0x1FAF8u}, {0x20000u, 0x2FFFDu}, {0x30000u, 0x3FFFDu}
	};
	return codepoint_in_ranges(codePoint, kWideRanges);
}
#endif

FASTAWC_FORCEINLINE uint64_t unicode_display_width(const uint32_t codePoint) noexcept {
	if (codePoint == '\t') {
		return 0;
	}
	if (codePoint < 0x20u || (codePoint >= 0x7Fu && codePoint < 0xA0u)) {
		return 0;
	}
	if (codePoint < 0x7Fu) {
		return 1;
	}

#ifdef _WIN32
	if (codePoint < 0x0300u) {
		return 1;
	}
	if ((codePoint >= 0x0400u && codePoint <= 0x0482u) ||
		(codePoint >= 0x048Au && codePoint <= 0x052Fu)) {
		return 1;
	}
	if ((codePoint >= 0x4E00u && codePoint <= 0x9FFFu) ||
		(codePoint >= 0xAC00u && codePoint <= 0xD7A3u)) {
		return 2;
	}
	if (is_zero_width_codepoint(codePoint)) {
		return 0;
	}
	return is_wide_codepoint(codePoint) ? 2u : 1u;
#else
	const int width = ::wcwidth(static_cast<wchar_t>(codePoint));
	return width > 0 ? static_cast<uint64_t>(width) : 0u;
#endif
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void handle_invalid_byte(Counts& out, ScanState& st) noexcept {
	if constexpr (CountWords) {
		out.wordCount += static_cast<uint64_t>(st.prevSpaceBit);
		st.prevSpaceBit = 0;
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void handle_codepoint(const uint32_t codePoint, Counts& out, ScanState& st) noexcept {
	if constexpr (CountLines) {
		out.lineCount += static_cast<uint64_t>(codePoint == '\n');
	}

	if constexpr (CountWords) {
		const uint32_t isSpace = static_cast<uint32_t>(is_unicode_whitespace(codePoint));
		out.wordCount += static_cast<uint64_t>((isSpace ^ 1u) & st.prevSpaceBit);
		st.prevSpaceBit = isSpace;
	}

	if constexpr (CountChars) {
		++out.charCount;
	}

	if constexpr (CountMaxLine) {
		if (codePoint == '\n') {
			if (!st.sawNewline) {
				st.prefixLineLength = st.currentLineLength;
				st.sawNewline = true;
			}
			if (st.currentLineLength > out.maxLineLength) {
				out.maxLineLength = st.currentLineLength;
			}
			st.currentLineLength = 0;
			return;
		}

		if (codePoint == '\t') {
			st.currentLineLength += tab_advance(st.currentLineLength);
		}
		else {
			st.currentLineLength += unicode_display_width(codePoint);
		}
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_word_scalar_chunk(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	auto mark_nonspace = [&]() noexcept {
		if constexpr (CountWords) {
			out.wordCount += static_cast<uint64_t>(st.prevSpaceBit);
			st.prevSpaceBit = 0;
		}
	};

	size_t i = 0;
	while (i < size) {
		const uint8_t c = data[i++];
		if (c < 0x80u) {
			if constexpr (CountLines) {
				out.lineCount += static_cast<uint64_t>(c == '\n');
			}
			if constexpr (CountWords) {
				const uint32_t isSpace = static_cast<uint32_t>(is_space_ascii(c));
				out.wordCount += static_cast<uint64_t>((isSpace ^ 1u) & st.prevSpaceBit);
				st.prevSpaceBit = isSpace;
			}
			continue;
		}

		size_t extraBytes = 0;
		if (c >= 0xC2u && c <= 0xDFu) {
			extraBytes = 1;
		}
		else if (c >= 0xE0u && c <= 0xEFu) {
			extraBytes = 2;
		}
		else if (c >= 0xF0u && c <= 0xF4u) {
			extraBytes = 3;
		}
		else {
			mark_nonspace();
			continue;
		}

		if (i + extraBytes > size) {
			mark_nonspace();
			return;
		}

		bool valid = true;
		for (size_t j = 0; j < extraBytes; ++j) {
			valid &= is_utf8_continuation(data[i + j]);
		}
		if (!valid) {
			mark_nonspace();
			continue;
		}

		bool isSpace = false;
		if constexpr (CountWords) {
			if (c == 0xC2u) {
				const uint8_t t0 = data[i];
				isSpace = t0 == 0x85u || t0 == 0xA0u;
			}
			else if (c == 0xE1u) {
				isSpace = data[i] == 0x9Au && data[i + 1] == 0x80u;
			}
			else if (c == 0xE2u) {
				const uint8_t t0 = data[i];
				const uint8_t t1 = data[i + 1];
				isSpace =
					(t0 == 0x80u && (t1 <= 0x8Au || t1 == 0xA8u || t1 == 0xA9u || t1 == 0xAFu)) ||
					(t0 == 0x81u && (t1 == 0x9Fu || t1 == 0xA0u));
			}
			else if (c == 0xE3u) {
				isSpace = data[i] == 0x80u && data[i + 1] == 0x80u;
			}

			const uint32_t spaceBit = static_cast<uint32_t>(isSpace);
			out.wordCount += static_cast<uint64_t>((spaceBit ^ 1u) & st.prevSpaceBit);
			st.prevSpaceBit = spaceBit;
		}

		i += extraBytes;
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_ascii_scalar_chunk(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	for (size_t i = 0; i < size; ++i) {
		const uint8_t c = data[i];

		if constexpr (CountLines) {
			out.lineCount += static_cast<uint64_t>(c == '\n');
		}

		if constexpr (CountWords) {
			const uint32_t isSpace = static_cast<uint32_t>(is_space_ascii(c));
			out.wordCount += static_cast<uint64_t>((isSpace ^ 1u) & st.prevSpaceBit);
			st.prevSpaceBit = isSpace;
		}

		if constexpr (CountChars) {
			++out.charCount;
		}

		if constexpr (CountMaxLine) {
			if (c == '\n') {
				if (!st.sawNewline) {
					st.prefixLineLength = st.currentLineLength;
					st.sawNewline = true;
				}
				if (st.currentLineLength > out.maxLineLength) {
					out.maxLineLength = st.currentLineLength;
				}
				st.currentLineLength = 0;
			}
			else if (c == '\t') {
				st.currentLineLength += tab_advance(st.currentLineLength);
			}
			else if (c >= 0x20u && c != 0x7Fu) {
				++st.currentLineLength;
			}
		}
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_charline_scalar_chunk(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	size_t i = 0;
	while (i < size) {
		if (st.utf8Expected == 0) {
			const size_t asciiBegin = i;
			while (i < size && data[i] < 0x80u) {
				++i;
			}
			if (i != asciiBegin) {
				process_strict_ascii_scalar_chunk<CountLines, false, CountChars, CountMaxLine>(
					data + asciiBegin, i - asciiBegin, out, st);
				if (i == size) {
					break;
				}
			}

			const uint8_t c = data[i++];
			if (c < 0x80u) {
				handle_codepoint<CountLines, false, CountChars, CountMaxLine>(c, out, st);
				continue;
			}
			if (c >= 0xC2u && c <= 0xDFu) {
				st.utf8CodePoint = c & 0x1Fu;
				st.utf8MinCodePoint = 0x80u;
				st.utf8Expected = 1;
				st.utf8Seen = 0;
				continue;
			}
			if (c >= 0xE0u && c <= 0xEFu) {
				st.utf8CodePoint = c & 0x0Fu;
				st.utf8MinCodePoint = 0x800u;
				st.utf8Expected = 2;
				st.utf8Seen = 0;
				continue;
			}
			if (c >= 0xF0u && c <= 0xF4u) {
				st.utf8CodePoint = c & 0x07u;
				st.utf8MinCodePoint = 0x10000u;
				st.utf8Expected = 3;
				st.utf8Seen = 0;
				continue;
			}

			handle_invalid_byte<CountLines, false, CountChars, CountMaxLine>(out, st);
			continue;
		}

		const uint8_t c = data[i++];
		if (is_utf8_continuation(c)) {
			st.utf8CodePoint = (st.utf8CodePoint << 6) | static_cast<uint32_t>(c & 0x3Fu);
			++st.utf8Seen;
			if (st.utf8Seen != st.utf8Expected) {
				continue;
			}

			const uint32_t codePoint = st.utf8CodePoint;
			const bool valid =
				codePoint >= st.utf8MinCodePoint &&
				codePoint <= 0x10FFFFu &&
				!(codePoint >= 0xD800u && codePoint <= 0xDFFFu);
			st.utf8CodePoint = 0;
			st.utf8MinCodePoint = 0;
			st.utf8Expected = 0;
			st.utf8Seen = 0;

			if (valid) {
				handle_codepoint<CountLines, false, CountChars, CountMaxLine>(codePoint, out, st);
			}
			else {
				handle_invalid_byte<CountLines, false, CountChars, CountMaxLine>(out, st);
			}
			continue;
		}

		st.utf8CodePoint = 0;
		st.utf8MinCodePoint = 0;
		st.utf8Expected = 0;
		st.utf8Seen = 0;
		handle_invalid_byte<CountLines, false, CountChars, CountMaxLine>(out, st);
		--i;
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_generic_scalar_chunk(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	size_t i = 0;
	while (i < size) {
		if (st.utf8Expected == 0) {
			const size_t asciiBegin = i;
			while (i < size && data[i] < 0x80u) {
				++i;
			}
			if (i != asciiBegin) {
				process_strict_ascii_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(
					data + asciiBegin, i - asciiBegin, out, st);
				if (i == size) {
					break;
				}
			}

			const uint8_t c = data[i++];
			if (c < 0x80u) {
				handle_codepoint<CountLines, CountWords, CountChars, CountMaxLine>(c, out, st);
				continue;
			}
			if (c >= 0xC2u && c <= 0xDFu) {
				st.utf8CodePoint = c & 0x1Fu;
				st.utf8MinCodePoint = 0x80u;
				st.utf8Expected = 1;
				st.utf8Seen = 0;
				continue;
			}
			if (c >= 0xE0u && c <= 0xEFu) {
				st.utf8CodePoint = c & 0x0Fu;
				st.utf8MinCodePoint = 0x800u;
				st.utf8Expected = 2;
				st.utf8Seen = 0;
				continue;
			}
			if (c >= 0xF0u && c <= 0xF4u) {
				st.utf8CodePoint = c & 0x07u;
				st.utf8MinCodePoint = 0x10000u;
				st.utf8Expected = 3;
				st.utf8Seen = 0;
				continue;
			}

			handle_invalid_byte<CountLines, CountWords, CountChars, CountMaxLine>(out, st);
			continue;
		}

		const uint8_t c = data[i++];
		if (is_utf8_continuation(c)) {
			st.utf8CodePoint = (st.utf8CodePoint << 6) | static_cast<uint32_t>(c & 0x3Fu);
			++st.utf8Seen;
			if (st.utf8Seen != st.utf8Expected) {
				continue;
			}

			const uint32_t codePoint = st.utf8CodePoint;
			const bool valid =
				codePoint >= st.utf8MinCodePoint &&
				codePoint <= 0x10FFFFu &&
				!(codePoint >= 0xD800u && codePoint <= 0xDFFFu);
			st.utf8CodePoint = 0;
			st.utf8MinCodePoint = 0;
			st.utf8Expected = 0;
			st.utf8Seen = 0;

			if (valid) {
				handle_codepoint<CountLines, CountWords, CountChars, CountMaxLine>(codePoint, out, st);
			}
			else {
				handle_invalid_byte<CountLines, CountWords, CountChars, CountMaxLine>(out, st);
			}
			continue;
		}

		st.utf8CodePoint = 0;
		st.utf8MinCodePoint = 0;
		st.utf8Expected = 0;
		st.utf8Seen = 0;
		handle_invalid_byte<CountLines, CountWords, CountChars, CountMaxLine>(out, st);
		--i;
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_fast_scalar_chunk(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	for (size_t i = 0; i < size; ++i) {
		const uint8_t c = data[i];

		if constexpr (CountLines) {
			out.lineCount += static_cast<uint64_t>(c == '\n');
		}

		if constexpr (CountWords) {
			const uint32_t isSpace = static_cast<uint32_t>(is_space_ascii(c));
			out.wordCount += static_cast<uint64_t>((isSpace ^ 1u) & st.prevSpaceBit);
			st.prevSpaceBit = isSpace;
		}

		uint32_t utf8Lead = 0;
		if constexpr (CountChars) {
			utf8Lead = is_utf8_lead(c);
			out.charCount += utf8Lead;
		}

		if constexpr (CountMaxLine) {
			if (c == '\n') {
				if (!st.sawNewline) {
					st.prefixLineLength = st.currentLineLength;
					st.sawNewline = true;
				}
				if (st.currentLineLength > out.maxLineLength) {
					out.maxLineLength = st.currentLineLength;
				}
				st.currentLineLength = 0;
			}
			else if constexpr (CountChars) {
				st.currentLineLength += utf8Lead;
			}
			else {
				++st.currentLineLength;
			}
		}
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_scalar_chunk(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	if constexpr (!CountChars && !CountMaxLine) {
		process_strict_word_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, size, out, st);
	}
	else if constexpr (!CountWords) {
		process_strict_charline_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, size, out, st);
	}
	else {
		process_strict_generic_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, size, out, st);
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void finalize_strict_scalar_state(Counts& out, ScanState& st) noexcept {
	if (st.utf8Expected == 0) {
		return;
	}
	st.utf8CodePoint = 0;
	st.utf8MinCodePoint = 0;
	st.utf8Expected = 0;
	st.utf8Seen = 0;
	handle_invalid_byte<CountLines, CountWords, CountChars, CountMaxLine>(out, st);
}

#if FASTAWC_BACKEND_ENABLE_AVX2
FASTAWC_FORCEINLINE __m256i nibble_lut_match(const __m256i v, const __m256i lowLut, const __m256i highLut) noexcept {
	const __m256i low = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
	const __m256i high = _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0F));
	const __m256i lowBits = _mm256_shuffle_epi8(lowLut, low);
	const __m256i highBits = _mm256_shuffle_epi8(highLut, high);
	return _mm256_and_si256(lowBits, highBits);
}

FASTAWC_FORCEINLINE uint32_t mask_newlines32(const __m256i v) noexcept {
	return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8('\n'))));
}

FASTAWC_FORCEINLINE bool is_ascii_block32(const __m256i v) noexcept {
	return _mm256_movemask_epi8(v) == 0;
}

FASTAWC_FORCEINLINE uint32_t mask_whitespace32(const __m256i v) noexcept {
	const __m256i lowLut = _mm256_setr_epi8(
		0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00,
		0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00, 0x00);
	const __m256i highLut = _mm256_setr_epi8(
		0x1F, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x1F, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
	const __m256i matches = nibble_lut_match(v, lowLut, highLut);
	const __m256i zero = _mm256_setzero_si256();
	const __m256i nonZero = _mm256_cmpeq_epi8(matches, zero);
	return ~static_cast<uint32_t>(_mm256_movemask_epi8(nonZero));
}

FASTAWC_FORCEINLINE uint32_t mask_utf8_lead32(const __m256i v) noexcept {
	const __m256i highNibbleLut = _mm256_setr_epi8(
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		static_cast<char>(0xFF), static_cast<char>(0xFF), static_cast<char>(0xFF), static_cast<char>(0xFF), 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		static_cast<char>(0xFF), static_cast<char>(0xFF), static_cast<char>(0xFF), static_cast<char>(0xFF), 0x00, 0x00, 0x00, 0x00);
	const __m256i high = _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0F));
	const __m256i continuation = _mm256_shuffle_epi8(highNibbleLut, high);
	const __m256i isLead = _mm256_cmpeq_epi8(continuation, _mm256_setzero_si256());
	return static_cast<uint32_t>(_mm256_movemask_epi8(isLead));
}

FASTAWC_FORCEINLINE uint32_t mask_special_display_ascii32(const __m256i v) noexcept {
	const __m256i control = _mm256_cmpgt_epi8(_mm256_set1_epi8(0x20), v);
	const __m256i newline = _mm256_cmpeq_epi8(v, _mm256_set1_epi8('\n'));
	const __m256i del = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x7F));
	const __m256i special = _mm256_or_si256(_mm256_andnot_si256(newline, control), del);
	return static_cast<uint32_t>(_mm256_movemask_epi8(special));
}

FASTAWC_FORCEINLINE bool can_use_ascii_display_fast_path(const __m256i v) noexcept {
	return mask_special_display_ascii32(v) == 0;
}

FASTAWC_FORCEINLINE uint32_t mask_ascii_tabs32(const __m256i v) noexcept {
	return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8('\t'))));
}

FASTAWC_FORCEINLINE uint32_t mask_ascii_zero_width32(const __m256i v) noexcept {
	const __m256i control = _mm256_cmpgt_epi8(_mm256_set1_epi8(0x20), v);
	const __m256i newline = _mm256_cmpeq_epi8(v, _mm256_set1_epi8('\n'));
	const __m256i tab = _mm256_cmpeq_epi8(v, _mm256_set1_epi8('\t'));
	const __m256i del = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(0x7F));
	const __m256i zeroWidth = _mm256_or_si256(_mm256_andnot_si256(_mm256_or_si256(newline, tab), control), del);
	return static_cast<uint32_t>(_mm256_movemask_epi8(zeroWidth));
}

FASTAWC_FORCEINLINE uint32_t mask_bytes_ge_f032(const __m256i v) noexcept {
	return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpgt_epi8(v, _mm256_set1_epi8(static_cast<char>(0xEF)))));
}

FASTAWC_FORCEINLINE bool can_use_strict_ascii_display_fast_path(const __m256i v) noexcept {
	return mask_ascii_tabs32(v) == 0;
}

FASTAWC_FORCEINLINE void update_max_line_from_masks(
	const uint32_t newlineMask,
	const uint32_t countMask,
	Counts& out,
	ScanState& st) noexcept
{
	if (newlineMask == 0) {
		st.currentLineLength += std::popcount(countMask);
		return;
	}

	unsigned segmentStart = 0;
	uint64_t carried = st.currentLineLength;
	uint32_t pending = newlineMask;
	while (pending != 0) {
		const unsigned newlinePos = std::countr_zero(pending);
		const uint32_t segmentMask = range_mask32(segmentStart, newlinePos);
		const uint64_t segmentLength = std::popcount(countMask & segmentMask);
		const uint64_t lineLength = carried + segmentLength;
		if (!st.sawNewline) {
			st.prefixLineLength = lineLength;
			st.sawNewline = true;
		}
		if (lineLength > out.maxLineLength) {
			out.maxLineLength = lineLength;
		}
		carried = 0;
		segmentStart = newlinePos + 1;
		pending &= pending - 1;
	}

	carried += std::popcount(countMask & range_mask32(segmentStart, 32));
	st.currentLineLength = carried;
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_fast_avx2_ascii_block(const __m256i block, Counts& out, ScanState& st) noexcept {
	uint32_t newlineMask = 0;
	if constexpr (CountLines || CountMaxLine) {
		newlineMask = mask_newlines32(block);
	}
	if constexpr (CountLines) {
		out.lineCount += std::popcount(newlineMask);
	}
	if constexpr (CountWords) {
		const uint32_t whitespaceMask = mask_whitespace32(block);
		const uint32_t startMask = ~whitespaceMask & ((whitespaceMask << 1) | st.prevSpaceBit);
		out.wordCount += std::popcount(startMask);
		st.prevSpaceBit = whitespaceMask >> 31;
	}
	if constexpr (CountChars) {
		out.charCount += 32;
	}
	if constexpr (CountMaxLine) {
		update_max_line_from_masks(newlineMask, ~newlineMask, out, st);
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_fast_avx2_generic_block(const __m256i block, Counts& out, ScanState& st) noexcept {
	uint32_t newlineMask = 0;
	if constexpr (CountLines || CountMaxLine) {
		newlineMask = mask_newlines32(block);
	}
	if constexpr (CountLines) {
		out.lineCount += std::popcount(newlineMask);
	}
	if constexpr (CountWords) {
		const uint32_t whitespaceMask = mask_whitespace32(block);
		const uint32_t startMask = ~whitespaceMask & ((whitespaceMask << 1) | st.prevSpaceBit);
		out.wordCount += std::popcount(startMask);
		st.prevSpaceBit = whitespaceMask >> 31;
	}
	if constexpr (CountChars || CountMaxLine) {
		const uint32_t utf8LeadMask = mask_utf8_lead32(block);
		if constexpr (CountChars) {
			out.charCount += std::popcount(utf8LeadMask);
		}
		if constexpr (CountMaxLine) {
			const uint32_t countMask = CountChars ? (utf8LeadMask & ~newlineMask) : ~newlineMask;
			update_max_line_from_masks(newlineMask, countMask, out, st);
		}
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_fast_avx2_word_block(const __m256i block, Counts& out, ScanState& st) noexcept {
	uint32_t newlineMask = 0;
	if constexpr (CountLines) {
		newlineMask = mask_newlines32(block);
		out.lineCount += std::popcount(newlineMask);
	}
	if constexpr (CountWords) {
		const uint32_t whitespaceMask = mask_whitespace32(block);
		const uint32_t startMask = ~whitespaceMask & ((whitespaceMask << 1) | st.prevSpaceBit);
		out.wordCount += std::popcount(startMask);
		st.prevSpaceBit = whitespaceMask >> 31;
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_fast_avx2_chunk(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	static constexpr size_t kPrefetchDistance = 512;
	size_t offset = 0;
	if constexpr (!CountChars && !CountMaxLine) {
		for (; offset + 128 <= size; offset += 128) {
			if (offset + kPrefetchDistance < size) {
				prefetch_read(data + offset + kPrefetchDistance);
			}
			process_fast_avx2_word_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 0)), out, st);
			process_fast_avx2_word_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 32)), out, st);
			process_fast_avx2_word_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 64)), out, st);
			process_fast_avx2_word_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 96)), out, st);
		}
		for (; offset + 32 <= size; offset += 32) {
			process_fast_avx2_word_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset)), out, st);
		}
	} else {
		for (; offset + 128 <= size; offset += 128) {
			if (offset + kPrefetchDistance < size) {
				prefetch_read(data + offset + kPrefetchDistance);
			}
			const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 0));
			const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 32));
			const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 64));
			const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 96));
			const __m256i ab = _mm256_or_si256(a, b);
			const __m256i cd = _mm256_or_si256(c, d);
			if (is_ascii_block32(_mm256_or_si256(ab, cd))) {
				process_fast_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(a, out, st);
				process_fast_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(b, out, st);
				process_fast_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(c, out, st);
				process_fast_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(d, out, st);
			} else {
				process_fast_avx2_generic_block<CountLines, CountWords, CountChars, CountMaxLine>(a, out, st);
				process_fast_avx2_generic_block<CountLines, CountWords, CountChars, CountMaxLine>(b, out, st);
				process_fast_avx2_generic_block<CountLines, CountWords, CountChars, CountMaxLine>(c, out, st);
				process_fast_avx2_generic_block<CountLines, CountWords, CountChars, CountMaxLine>(d, out, st);
			}
		}
		for (; offset + 32 <= size; offset += 32) {
			const __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset));
			if (is_ascii_block32(block)) {
				process_fast_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(block, out, st);
			} else {
				process_fast_avx2_generic_block<CountLines, CountWords, CountChars, CountMaxLine>(block, out, st);
			}
		}
	}
	process_fast_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data + offset, size - offset, out, st);
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_avx2_scalar_block(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	process_strict_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, size, out, st);
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_avx2_ascii_block(const __m256i block, Counts& out, ScanState& st) noexcept {
	uint32_t newlineMask = 0;
	if constexpr (CountLines || CountMaxLine) {
		newlineMask = mask_newlines32(block);
	}
	if constexpr (CountLines) {
		out.lineCount += std::popcount(newlineMask);
	}
	if constexpr (CountWords) {
		const uint32_t whitespaceMask = mask_whitespace32(block);
		const uint32_t startMask = ~whitespaceMask & ((whitespaceMask << 1) | st.prevSpaceBit);
		out.wordCount += std::popcount(startMask);
		st.prevSpaceBit = whitespaceMask >> 31;
	}
	if constexpr (CountChars) {
		out.charCount += 32;
	}
	if constexpr (CountMaxLine) {
		const uint32_t zeroWidthMask = mask_ascii_zero_width32(block);
		const uint32_t countMask = ~(newlineMask | zeroWidthMask);
		update_max_line_from_masks(newlineMask, countMask, out, st);
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE bool try_process_strict_short_utf8_mixed_block(
	const uint8_t* const data,
	const uint32_t nonAsciiMask,
	Counts& out,
	ScanState& st) noexcept
{
	if (st.utf8Expected != 0) {
		return false;
	}

	unsigned position = 0;
	while (position < 32) {
		const uint32_t bit = 1u << position;
		if ((nonAsciiMask & bit) == 0) {
			const unsigned asciiBegin = position;
			++position;
			while (position < 32 && ((nonAsciiMask & (1u << position)) == 0)) {
				++position;
			}
			process_strict_ascii_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(
				data + asciiBegin,
				position - asciiBegin,
				out,
				st);
			continue;
		}

		if (is_utf8_continuation(data[position]) || position + 1 >= 32) {
			return false;
		}

		const uint8_t lead = data[position];
		uint32_t codePoint = 0;
		if (lead >= 0xC2u && lead <= 0xDFu) {
			const uint8_t continuation = data[position + 1];
			if (!is_utf8_continuation(continuation)) {
				return false;
			}
			codePoint =
				(static_cast<uint32_t>(lead & 0x1Fu) << 6) |
				static_cast<uint32_t>(continuation & 0x3Fu);
			position += 2;
		}
		else if (lead >= 0xE0u && lead <= 0xEFu) {
			if (position + 2 >= 32) {
				return false;
			}

			const uint8_t continuation0 = data[position + 1];
			const uint8_t continuation1 = data[position + 2];
			if (!is_utf8_continuation(continuation0) || !is_utf8_continuation(continuation1)) {
				return false;
			}
			if ((lead == 0xE0u && continuation0 < 0xA0u) ||
				(lead == 0xEDu && continuation0 >= 0xA0u)) {
				return false;
			}

			codePoint =
				(static_cast<uint32_t>(lead & 0x0Fu) << 12) |
				(static_cast<uint32_t>(continuation0 & 0x3Fu) << 6) |
				static_cast<uint32_t>(continuation1 & 0x3Fu);
			position += 3;
		}
		else {
			return false;
		}

		handle_codepoint<CountLines, CountWords, CountChars, CountMaxLine>(codePoint, out, st);
	}

	return true;
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_avx2_dispatch_block(
	const __m256i block,
	const uint8_t* const data,
	Counts& out,
	ScanState& st) noexcept
{
	const uint32_t nonAsciiMask = static_cast<uint32_t>(_mm256_movemask_epi8(block));
	if (nonAsciiMask == 0) {
		if constexpr (!CountMaxLine) {
			process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(block, out, st);
		}
		else if (can_use_strict_ascii_display_fast_path(block)) {
			process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(block, out, st);
		}
		else {
			process_strict_ascii_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, 32, out, st);
		}
		return;
	}

	if constexpr (CountChars || CountMaxLine) {
		if (mask_bytes_ge_f032(block) == 0 &&
			try_process_strict_short_utf8_mixed_block<CountLines, CountWords, CountChars, CountMaxLine>(
				data,
				nonAsciiMask,
				out,
				st)) {
			return;
		}
	}

	unsigned position = 0;
	while (position < 32) {
		const uint32_t shifted = nonAsciiMask >> position;
		if (shifted == 0) {
			process_strict_ascii_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(
				data + position,
				32 - position,
				out,
				st);
			return;
		}

		const unsigned asciiLength = std::countr_zero(shifted);
		if (asciiLength != 0) {
			process_strict_ascii_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(
				data + position,
				asciiLength,
				out,
				st);
			position += asciiLength;
		}

		const uint32_t nonAsciiShifted = nonAsciiMask >> position;
		const unsigned nonAsciiLength = std::countr_zero(~nonAsciiShifted);
		const unsigned spanLength = std::min<unsigned>(nonAsciiLength, 32 - position);
		process_strict_avx2_scalar_block<CountLines, CountWords, CountChars, CountMaxLine>(
			data + position,
			spanLength,
			out,
			st);
		position += spanLength;
	}
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
FASTAWC_FORCEINLINE void process_strict_avx2_chunk(
	const uint8_t* const data,
	const size_t size,
	Counts& out,
	ScanState& st) noexcept
{
	static constexpr size_t kPrefetchDistance = 512;
	size_t offset = 0;

	if constexpr (!CountWords && !CountChars && !CountMaxLine) {
		for (; offset + 128 <= size; offset += 128) {
			if (offset + kPrefetchDistance < size) {
				prefetch_read(data + offset + kPrefetchDistance);
			}
			process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 0)), out, st);
			process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 32)), out, st);
			process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 64)), out, st);
			process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 96)), out, st);
		}
		for (; offset + 32 <= size; offset += 32) {
			process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset)), out, st);
		}
	} else {
		for (; offset + 128 <= size; offset += 128) {
			if (offset + kPrefetchDistance < size) {
				prefetch_read(data + offset + kPrefetchDistance);
			}
			const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 0));
			const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 32));
			const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 64));
			const __m256i d = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 96));
			const __m256i ab = _mm256_or_si256(a, b);
			const __m256i cd = _mm256_or_si256(c, d);
			if (is_ascii_block32(_mm256_or_si256(ab, cd))) {
				if constexpr (!CountMaxLine) {
					process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(a, out, st);
					process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(b, out, st);
					process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(c, out, st);
					process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(d, out, st);
				}
				else {
					if (can_use_strict_ascii_display_fast_path(a) &&
						can_use_strict_ascii_display_fast_path(b) &&
						can_use_strict_ascii_display_fast_path(c) &&
						can_use_strict_ascii_display_fast_path(d)) {
						process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(a, out, st);
						process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(b, out, st);
						process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(c, out, st);
						process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(d, out, st);
					}
					else {
						process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(a, data + offset + 0, out, st);
						process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(b, data + offset + 32, out, st);
						process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(c, data + offset + 64, out, st);
						process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(d, data + offset + 96, out, st);
					}
				}
			}
			else {
				process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(a, data + offset + 0, out, st);
				process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(b, data + offset + 32, out, st);
				process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(c, data + offset + 64, out, st);
				process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(d, data + offset + 96, out, st);
			}
		}
		for (; offset + 32 <= size; offset += 32) {
			process_strict_avx2_dispatch_block<CountLines, CountWords, CountChars, CountMaxLine>(
				_mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset)),
				data + offset,
				out,
				st);
		}
	}

	process_strict_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data + offset, size - offset, out, st);
}
#endif

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
void process_fast_chunk(const uint8_t* const data, const size_t size, Counts& out, ScanState& st) noexcept {
#if FASTAWC_BACKEND_ENABLE_AVX2
	process_fast_avx2_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, size, out, st);
#else
	process_fast_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, size, out, st);
#endif
}

template<bool CountLines, bool CountWords, bool CountChars, bool CountMaxLine>
void process_strict_chunk(const uint8_t* const data, const size_t size, Counts& out, ScanState& st) noexcept {
#if FASTAWC_BACKEND_ENABLE_AVX2
	process_strict_avx2_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, size, out, st);
#else
	process_strict_scalar_chunk<CountLines, CountWords, CountChars, CountMaxLine>(data, size, out, st);
#endif
}

ScanProcessor select_fast_processor(const uint32_t scanMode) noexcept {
	static constexpr ScanProcessor kDispatchTable[16] = {
		nullptr,
		&process_fast_chunk<true, false, false, false>,
		&process_fast_chunk<false, true, false, false>,
		&process_fast_chunk<true, true, false, false>,
		&process_fast_chunk<false, false, true, false>,
		&process_fast_chunk<true, false, true, false>,
		&process_fast_chunk<false, true, true, false>,
		&process_fast_chunk<true, true, true, false>,
		&process_fast_chunk<false, false, false, true>,
		&process_fast_chunk<true, false, false, true>,
		&process_fast_chunk<false, true, false, true>,
		&process_fast_chunk<true, true, false, true>,
		&process_fast_chunk<false, false, true, true>,
		&process_fast_chunk<true, false, true, true>,
		&process_fast_chunk<false, true, true, true>,
		&process_fast_chunk<true, true, true, true>
	};
	return scanMode < 16u ? kDispatchTable[scanMode] : nullptr;
}

ScanProcessor select_strict_processor(const uint32_t scanMode) noexcept {
	static constexpr ScanProcessor kDispatchTable[16] = {
		nullptr,
		&process_strict_chunk<true, false, false, false>,
		&process_strict_chunk<false, true, false, false>,
		&process_strict_chunk<true, true, false, false>,
		&process_strict_chunk<false, false, true, false>,
		&process_strict_chunk<true, false, true, false>,
		&process_strict_chunk<false, true, true, false>,
		&process_strict_chunk<true, true, true, false>,
		&process_strict_chunk<false, false, false, true>,
		&process_strict_chunk<true, false, false, true>,
		&process_strict_chunk<false, true, false, true>,
		&process_strict_chunk<true, true, false, true>,
		&process_strict_chunk<false, false, true, true>,
		&process_strict_chunk<true, false, true, true>,
		&process_strict_chunk<false, true, true, true>,
		&process_strict_chunk<true, true, true, true>
	};
	return scanMode < 16u ? kDispatchTable[scanMode] : nullptr;
}

const Backend& backend_instance() noexcept {
	static const Backend backend{
		FASTAWC_BACKEND_NAME,
		FASTAWC_BACKEND_ENABLE_AVX2 != 0,
		&select_fast_processor,
		&select_strict_processor
	};
	return backend;
}

} // namespace FASTAWC_BACKEND_NAMESPACE
} // namespace fastawc
