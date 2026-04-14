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
FASTAWC_FORCEINLINE bool is_combining_mark(const uint32_t codePoint) noexcept {
	return
		(codePoint >= 0x0300u && codePoint <= 0x036Fu) ||
		(codePoint >= 0x0483u && codePoint <= 0x0489u) ||
		(codePoint >= 0x0591u && codePoint <= 0x05BDu) ||
		codePoint == 0x05BFu ||
		(codePoint >= 0x05C1u && codePoint <= 0x05C2u) ||
		(codePoint >= 0x05C4u && codePoint <= 0x05C5u) ||
		codePoint == 0x05C7u ||
		(codePoint >= 0x0610u && codePoint <= 0x061Au) ||
		(codePoint >= 0x064Bu && codePoint <= 0x065Fu) ||
		codePoint == 0x0670u ||
		(codePoint >= 0x06D6u && codePoint <= 0x06DDu) ||
		(codePoint >= 0x06DFu && codePoint <= 0x06E4u) ||
		(codePoint >= 0x06E7u && codePoint <= 0x06E8u) ||
		(codePoint >= 0x06EAu && codePoint <= 0x06EDu) ||
		codePoint == 0x0711u ||
		(codePoint >= 0x0730u && codePoint <= 0x074Au) ||
		(codePoint >= 0x07A6u && codePoint <= 0x07B0u) ||
		(codePoint >= 0x07EBu && codePoint <= 0x07F3u) ||
		(codePoint >= 0x0816u && codePoint <= 0x0819u) ||
		(codePoint >= 0x081Bu && codePoint <= 0x0823u) ||
		(codePoint >= 0x0825u && codePoint <= 0x0827u) ||
		(codePoint >= 0x0829u && codePoint <= 0x082Du) ||
		(codePoint >= 0x0859u && codePoint <= 0x085Bu) ||
		(codePoint >= 0x08D3u && codePoint <= 0x08FFu) ||
		(codePoint >= 0x20D0u && codePoint <= 0x20FFu) ||
		(codePoint >= 0xFE20u && codePoint <= 0xFE2Fu);
}

FASTAWC_FORCEINLINE bool is_wide_codepoint(const uint32_t codePoint) noexcept {
	return
		(codePoint >= 0x1100u && codePoint <= 0x115Fu) ||
		codePoint == 0x2329u || codePoint == 0x232Au ||
		(codePoint >= 0x2E80u && codePoint <= 0xA4CFu && codePoint != 0x303Fu) ||
		(codePoint >= 0xAC00u && codePoint <= 0xD7A3u) ||
		(codePoint >= 0xF900u && codePoint <= 0xFAFFu) ||
		(codePoint >= 0xFE10u && codePoint <= 0xFE19u) ||
		(codePoint >= 0xFE30u && codePoint <= 0xFE6Fu) ||
		(codePoint >= 0xFF00u && codePoint <= 0xFF60u) ||
		(codePoint >= 0xFFE0u && codePoint <= 0xFFE6u) ||
		(codePoint >= 0x1F300u && codePoint <= 0x1FAFFu) ||
		(codePoint >= 0x20000u && codePoint <= 0x3FFFDu);
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
	if (is_combining_mark(codePoint)) {
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
	size_t i = 0;
	while (i < size) {
		const uint8_t c = data[i++];

		if (st.utf8Expected == 0) {
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
	process_fast_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(block, out, st);
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
			const bool allAscii = is_ascii_block32(_mm256_or_si256(ab, cd));
			if (allAscii) {
				if constexpr (CountMaxLine) {
					if (can_use_ascii_display_fast_path(a) &&
						can_use_ascii_display_fast_path(b) &&
						can_use_ascii_display_fast_path(c) &&
						can_use_ascii_display_fast_path(d)) {
						process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(a, out, st);
						process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(b, out, st);
						process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(c, out, st);
						process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(d, out, st);
					} else {
						process_strict_avx2_scalar_block<CountLines, CountWords, CountChars, CountMaxLine>(data + offset, 128, out, st);
					}
				} else {
					process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(a, out, st);
					process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(b, out, st);
					process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(c, out, st);
					process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(d, out, st);
				}
			} else {
				process_strict_avx2_scalar_block<CountLines, CountWords, CountChars, CountMaxLine>(data + offset, 128, out, st);
			}
		}
		for (; offset + 32 <= size; offset += 32) {
			const __m256i block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset));
			if (is_ascii_block32(block) && (!CountMaxLine || can_use_ascii_display_fast_path(block))) {
				process_strict_avx2_ascii_block<CountLines, CountWords, CountChars, CountMaxLine>(block, out, st);
			} else {
				process_strict_avx2_scalar_block<CountLines, CountWords, CountChars, CountMaxLine>(data + offset, 32, out, st);
			}
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
	switch (scanMode) {
	case 0x0u: return nullptr;
	case 0x1u: return &process_fast_chunk<true, false, false, false>;
	case 0x2u: return &process_fast_chunk<false, true, false, false>;
	case 0x3u: return &process_fast_chunk<true, true, false, false>;
	case 0x4u: return &process_fast_chunk<false, false, true, false>;
	case 0x5u: return &process_fast_chunk<true, false, true, false>;
	case 0x6u: return &process_fast_chunk<false, true, true, false>;
	case 0x7u: return &process_fast_chunk<true, true, true, false>;
	case 0x8u: return &process_fast_chunk<false, false, false, true>;
	case 0x9u: return &process_fast_chunk<true, false, false, true>;
	case 0xAu: return &process_fast_chunk<false, true, false, true>;
	case 0xBu: return &process_fast_chunk<true, true, false, true>;
	case 0xCu: return &process_fast_chunk<false, false, true, true>;
	case 0xDu: return &process_fast_chunk<true, false, true, true>;
	case 0xEu: return &process_fast_chunk<false, true, true, true>;
	case 0xFu: return &process_fast_chunk<true, true, true, true>;
	default: return nullptr;
	}
}

ScanProcessor select_strict_processor(const uint32_t scanMode) noexcept {
	switch (scanMode) {
	case 0x0u: return nullptr;
	case 0x1u: return &process_strict_chunk<true, false, false, false>;
	case 0x2u: return &process_strict_chunk<false, true, false, false>;
	case 0x3u: return &process_strict_chunk<true, true, false, false>;
	case 0x4u: return &process_strict_chunk<false, false, true, false>;
	case 0x5u: return &process_strict_chunk<true, false, true, false>;
	case 0x6u: return &process_strict_chunk<false, true, true, false>;
	case 0x7u: return &process_strict_chunk<true, true, true, false>;
	case 0x8u: return &process_strict_chunk<false, false, false, true>;
	case 0x9u: return &process_strict_chunk<true, false, false, true>;
	case 0xAu: return &process_strict_chunk<false, true, false, true>;
	case 0xBu: return &process_strict_chunk<true, true, false, true>;
	case 0xCu: return &process_strict_chunk<false, false, true, true>;
	case 0xDu: return &process_strict_chunk<true, false, true, true>;
	case 0xEu: return &process_strict_chunk<false, true, true, true>;
	case 0xFu: return &process_strict_chunk<true, true, true, true>;
	default: return nullptr;
	}
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
