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

#include "engine_unicode.h"

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

FASTAWC_FORCEINLINE uint32_t mask_byte_range32(const __m256i v, const uint8_t lo, const uint8_t hi) noexcept {
	const __m256i low = _mm256_set1_epi8(static_cast<char>(lo));
	const __m256i high = _mm256_set1_epi8(static_cast<char>(hi));
	const __m256i geLow = _mm256_cmpeq_epi8(_mm256_max_epu8(v, low), v);
	const __m256i leHigh = _mm256_cmpeq_epi8(_mm256_min_epu8(v, high), v);
	return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_and_si256(geLow, leHigh)));
}

FASTAWC_FORCEINLINE uint32_t mask_whitespace32(const __m256i v) noexcept {
	const __m256i space = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(' '));
	const uint32_t controlMask = mask_byte_range32(v, '\t', '\r');
	const uint32_t spaceMask = static_cast<uint32_t>(_mm256_movemask_epi8(space));
	return spaceMask | controlMask;
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
	return mask_byte_range32(v, 0xF0u, 0xFFu);
}

struct Utf8BlockStructure {
	uint32_t continuationMask = 0;
	uint32_t lead2Mask = 0;
	uint32_t lead3Mask = 0;
	uint32_t lead4Mask = 0;
	uint32_t invalidMask = 0;
	uint32_t invalidSecondMask = 0;
};

FASTAWC_FORCEINLINE uint32_t mask_byte_eq32(const __m256i v, const uint8_t value) noexcept {
	return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, _mm256_set1_epi8(static_cast<char>(value)))));
}

FASTAWC_FORCEINLINE Utf8BlockStructure classify_utf8_block32(
	const __m256i v,
	const uint32_t nonAsciiMask) noexcept
{
	Utf8BlockStructure structure{};
	structure.continuationMask = mask_byte_range32(v, 0x80u, 0xBFu);
	structure.lead2Mask = mask_byte_range32(v, 0xC2u, 0xDFu);
	structure.lead3Mask = mask_byte_range32(v, 0xE0u, 0xEFu);
	structure.lead4Mask = mask_byte_range32(v, 0xF0u, 0xF4u);
	structure.invalidMask =
		nonAsciiMask & ~(structure.continuationMask | structure.lead2Mask | structure.lead3Mask | structure.lead4Mask);

	const uint32_t e0Second = mask_byte_eq32(v, 0xE0u) << 1;
	const uint32_t edSecond = mask_byte_eq32(v, 0xEDu) << 1;
	const uint32_t f0Second = mask_byte_eq32(v, 0xF0u) << 1;
	const uint32_t f4Second = mask_byte_eq32(v, 0xF4u) << 1;
	structure.invalidSecondMask =
		(e0Second & ~mask_byte_range32(v, 0xA0u, 0xBFu)) |
		(edSecond & ~mask_byte_range32(v, 0x80u, 0x9Fu)) |
		(f0Second & ~mask_byte_range32(v, 0x90u, 0xBFu)) |
		(f4Second & ~mask_byte_range32(v, 0x80u, 0x8Fu));
	return structure;
}

FASTAWC_FORCEINLINE bool can_try_process_strict_utf8_block(const Utf8BlockStructure& structure) noexcept {
	constexpr uint32_t kBit31 = 1u << 31;
	constexpr uint32_t kBits30To31 = 3u << 30;
	constexpr uint32_t kBits29To31 = 7u << 29;
	const uint32_t boundaryCrossing =
		(structure.lead2Mask & kBit31) |
		(structure.lead3Mask & kBits30To31) |
		(structure.lead4Mask & kBits29To31);
	const uint32_t expectedContinuationMask =
		((structure.lead2Mask | structure.lead3Mask | structure.lead4Mask) << 1) |
		((structure.lead3Mask | structure.lead4Mask) << 2) |
		(structure.lead4Mask << 3);
	return
		boundaryCrossing == 0 &&
		structure.invalidMask == 0 &&
		structure.invalidSecondMask == 0 &&
		expectedContinuationMask == structure.continuationMask;
}

FASTAWC_FORCEINLINE uint32_t mask_strict_unicode_space_leads32(const __m256i v) noexcept {
	const __m256i c2 = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(static_cast<char>(0xC2)));
	const __m256i e1 = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(static_cast<char>(0xE1)));
	const __m256i e2 = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(static_cast<char>(0xE2)));
	const __m256i e3 = _mm256_cmpeq_epi8(v, _mm256_set1_epi8(static_cast<char>(0xE3)));
	const __m256i c2e1 = _mm256_or_si256(c2, e1);
	const __m256i e2e3 = _mm256_or_si256(e2, e3);
	return static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_or_si256(c2e1, e2e3)));
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

FASTAWC_FORCEINLINE void update_strict_ascii_display_from_masks(
	const uint32_t newlineMask,
	const uint32_t tabMask,
	const uint32_t countMask,
	Counts& out,
	ScanState& st) noexcept
{
	if (tabMask == 0) {
		update_max_line_from_masks(newlineMask, countMask, out, st);
		return;
	}

	unsigned segmentStart = 0;
	uint64_t carried = st.currentLineLength;
	uint32_t pending = newlineMask | tabMask;
	while (pending != 0) {
		const unsigned specialPos = std::countr_zero(pending);
		const uint32_t segmentMask = range_mask32(segmentStart, specialPos);
		carried += std::popcount(countMask & segmentMask);

		const uint32_t specialBit = 1u << specialPos;
		if ((newlineMask & specialBit) != 0) {
			if (!st.sawNewline) {
				st.prefixLineLength = carried;
				st.sawNewline = true;
			}
			if (carried > out.maxLineLength) {
				out.maxLineLength = carried;
			}
			carried = 0;
		}
		else {
			carried += tab_advance(carried);
		}

		segmentStart = specialPos + 1;
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
FASTAWC_FORCEINLINE void process_strict_avx2_ascii_display_block(const __m256i block, Counts& out, ScanState& st) noexcept {
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
		const uint32_t tabMask = mask_ascii_tabs32(block);
		const uint32_t zeroWidthMask = mask_ascii_zero_width32(block);
		const uint32_t countMask = ~(newlineMask | tabMask | zeroWidthMask);
		update_strict_ascii_display_from_masks(newlineMask, tabMask, countMask, out, st);
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
		else if (lead >= 0xF0u && lead <= 0xF4u) {
			if (position + 3 >= 32) {
				return false;
			}

			const uint8_t continuation0 = data[position + 1];
			const uint8_t continuation1 = data[position + 2];
			const uint8_t continuation2 = data[position + 3];
			if (!is_utf8_continuation(continuation0) ||
				!is_utf8_continuation(continuation1) ||
				!is_utf8_continuation(continuation2)) {
				return false;
			}
			if ((lead == 0xF0u && continuation0 < 0x90u) ||
				(lead == 0xF4u && continuation0 >= 0x90u)) {
				return false;
			}

			codePoint =
				(static_cast<uint32_t>(lead & 0x07u) << 18) |
				(static_cast<uint32_t>(continuation0 & 0x3Fu) << 12) |
				(static_cast<uint32_t>(continuation1 & 0x3Fu) << 6) |
				static_cast<uint32_t>(continuation2 & 0x3Fu);
			position += 4;
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
			process_strict_avx2_ascii_display_block<CountLines, CountWords, CountChars, CountMaxLine>(block, out, st);
		}
		return;
	}

	if constexpr (CountChars || CountMaxLine) {
		if (mask_bytes_ge_f032(block) == 0) {
			if (nonAsciiMask == 0xFFFFFFFFu &&
				try_process_strict_short_utf8_mixed_block<CountLines, CountWords, CountChars, CountMaxLine>(
					data,
					nonAsciiMask,
					out,
					st)) {
				return;
			}
		}
		else {
			const Utf8BlockStructure structure = classify_utf8_block32(block, nonAsciiMask);
			if (st.utf8Expected == 0 &&
				can_try_process_strict_utf8_block(structure) &&
				try_process_strict_short_utf8_mixed_block<CountLines, CountWords, CountChars, CountMaxLine>(
					data,
					nonAsciiMask,
					out,
					st)) {
				return;
			}
		}
	}
	else if constexpr (CountWords) {
		if (mask_strict_unicode_space_leads32(block) == 0) {
			process_fast_avx2_word_block<CountLines, CountWords, CountChars, CountMaxLine>(block, out, st);
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

template<uint32_t ScanMode>
constexpr ScanProcessor make_fast_dispatch_entry() noexcept {
	if constexpr (ScanMode == 0) {
		return nullptr;
	}
	else {
		return &process_fast_chunk<
			(ScanMode & kScanLines) != 0,
			(ScanMode & kScanWords) != 0,
			(ScanMode & kScanChars) != 0,
			(ScanMode & kScanMaxLine) != 0>;
	}
}

template<uint32_t ScanMode>
constexpr ScanProcessor make_strict_dispatch_entry() noexcept {
	if constexpr (ScanMode == 0) {
		return nullptr;
	}
	else {
		return &process_strict_chunk<
			(ScanMode & kScanLines) != 0,
			(ScanMode & kScanWords) != 0,
			(ScanMode & kScanChars) != 0,
			(ScanMode & kScanMaxLine) != 0>;
	}
}

ScanProcessor select_fast_processor(const uint32_t scanMode) noexcept {
	static constexpr ScanProcessor kDispatchTable[16] = {
		make_fast_dispatch_entry<0>(),
		make_fast_dispatch_entry<1>(),
		make_fast_dispatch_entry<2>(),
		make_fast_dispatch_entry<3>(),
		make_fast_dispatch_entry<4>(),
		make_fast_dispatch_entry<5>(),
		make_fast_dispatch_entry<6>(),
		make_fast_dispatch_entry<7>(),
		make_fast_dispatch_entry<8>(),
		make_fast_dispatch_entry<9>(),
		make_fast_dispatch_entry<10>(),
		make_fast_dispatch_entry<11>(),
		make_fast_dispatch_entry<12>(),
		make_fast_dispatch_entry<13>(),
		make_fast_dispatch_entry<14>(),
		make_fast_dispatch_entry<15>()
	};
	static_assert(kDispatchTable[0] == nullptr);
	static_assert(kDispatchTable[15] == make_fast_dispatch_entry<15>());
	return scanMode < 16u ? kDispatchTable[scanMode] : nullptr;
}

ScanProcessor select_strict_processor(const uint32_t scanMode) noexcept {
	static constexpr ScanProcessor kDispatchTable[16] = {
		make_strict_dispatch_entry<0>(),
		make_strict_dispatch_entry<1>(),
		make_strict_dispatch_entry<2>(),
		make_strict_dispatch_entry<3>(),
		make_strict_dispatch_entry<4>(),
		make_strict_dispatch_entry<5>(),
		make_strict_dispatch_entry<6>(),
		make_strict_dispatch_entry<7>(),
		make_strict_dispatch_entry<8>(),
		make_strict_dispatch_entry<9>(),
		make_strict_dispatch_entry<10>(),
		make_strict_dispatch_entry<11>(),
		make_strict_dispatch_entry<12>(),
		make_strict_dispatch_entry<13>(),
		make_strict_dispatch_entry<14>(),
		make_strict_dispatch_entry<15>()
	};
	static_assert(kDispatchTable[0] == nullptr);
	static_assert(kDispatchTable[15] == make_strict_dispatch_entry<15>());
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
