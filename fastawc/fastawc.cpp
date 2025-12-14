#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

#ifdef __AVX2__
#include <immintrin.h>
#endif

struct Counts {
	uint64_t lineCount = 0;
	uint64_t wordCount = 0;
	uint64_t byteCount = 0;
	uint64_t charCount = 0;
	uint64_t maxLineLength = 0;
};

struct Options {
	bool optLines = false;
	bool optWords = false;
	bool optBytes = false;
	bool optChars = false;
	bool optMaxLine = false;
	std::vector<std::string> files;
};

static constexpr size_t kBufSize = 4u << 20;

alignas(32) static std::array<uint8_t, 256> gIsSpace{};
inline void initSpaceTable() {
	gIsSpace.fill(0);
	gIsSpace[' '] = 1;
	gIsSpace['\n'] = 1;
	gIsSpace['\t'] = 1;
	gIsSpace['\r'] = 1;
	gIsSpace['\v'] = 1;
	gIsSpace['\f'] = 1;
}
inline bool isSpaceAscii(unsigned char c) { return gIsSpace[c] != 0; }
inline bool isUtf8Lead(unsigned char c) { return (c & 0xC0) != 0x80; }

struct ScalarState {
	bool prevSpace = true;
	uint64_t currentLineLen = 0;
};

#ifdef __AVX2__
struct Avx2State {
	uint32_t prevSpaceBit = 1;
	uint64_t currentLineLen = 0;
};
#endif

inline void processScalar(const unsigned char* buf, size_t n, Counts& out, ScalarState& st,
	bool countLines, bool countWords, bool countBytes,
	bool countChars, bool countMaxLine)
{
	if (countBytes) out.byteCount += n;
	for (size_t i = 0; i < n; ++i) {
		unsigned char c = buf[i];
		if (countLines && c == '\n') out.lineCount++;
		bool space = isSpaceAscii(c);
		if (countWords) {
			if (!space && st.prevSpace) out.wordCount++;
		}
		st.prevSpace = space;
		if (countChars) {
			if (isUtf8Lead(c)) {
				out.charCount++;
				if (countMaxLine) st.currentLineLen++;
			}
		}
		else if (countMaxLine) {
			st.currentLineLen++;
		}
		if (countMaxLine && c == '\n') {
			if (st.currentLineLen > out.maxLineLength) out.maxLineLength = st.currentLineLen;
			st.currentLineLen = 0;
		}
	}
}

inline void finalizeScalar(Counts& out, ScalarState& st, bool countMaxLine) {
	if (countMaxLine && st.currentLineLen > out.maxLineLength)
		out.maxLineLength = st.currentLineLen;
}

#ifdef __AVX2__
inline __m256i vset1(uint8_t c) { return _mm256_set1_epi8((char)c); }
inline uint32_t maskNewlines32(const __m256i v) {
	__m256i cmp = _mm256_cmpeq_epi8(v, vset1('\n'));
	return (uint32_t)_mm256_movemask_epi8(cmp);
}
inline uint32_t maskWhitespace32(const __m256i v) {
	__m256i mSpace = _mm256_cmpeq_epi8(v, vset1(' '));
	__m256i mN = _mm256_cmpeq_epi8(v, vset1('\n'));
	__m256i mT = _mm256_cmpeq_epi8(v, vset1('\t'));
	__m256i mR = _mm256_cmpeq_epi8(v, vset1('\r'));
	__m256i mV = _mm256_cmpeq_epi8(v, vset1('\v'));
	__m256i mF = _mm256_cmpeq_epi8(v, vset1('\f'));
	__m256i or1 = _mm256_or_si256(mSpace, mN);
	__m256i or2 = _mm256_or_si256(mT, mR);
	__m256i or3 = _mm256_or_si256(mV, mF);
	__m256i or4 = _mm256_or_si256(or1, or2);
	__m256i ws = _mm256_or_si256(or4, or3);
	return (uint32_t)_mm256_movemask_epi8(ws);
}
inline uint32_t maskUtf8Lead32(const __m256i v) {
	__m256i top2 = _mm256_and_si256(v, _mm256_set1_epi8((char)0xC0));
	__m256i cmp = _mm256_cmpeq_epi8(top2, _mm256_set1_epi8((char)0x80));
	__m256i lead = _mm256_xor_si256(cmp, _mm256_set1_epi8((char)0xFF));
	return (uint32_t)_mm256_movemask_epi8(lead);
}
inline uint32_t popcnt32(uint32_t x) {
#if defined(_MSC_VER)
	return __popcnt(x);
#else
	return (uint32_t)__builtin_popcount(x);
#endif
}
inline void processBlock32(const __m256i v, Counts& out, Avx2State& st,
	bool countLines, bool countWords, bool countBytes,
	bool countChars, bool countMaxLine)
{
	uint32_t nl = maskNewlines32(v);
	if (countLines) out.lineCount += popcnt32(nl);
	if (countWords) {
		uint32_t ws = maskWhitespace32(v);
		uint32_t prevShift = (ws << 1) | st.prevSpaceBit;
		uint32_t startMask = (~ws) & prevShift;
		out.wordCount += popcnt32(startMask);
		st.prevSpaceBit = (ws >> 31) & 1u;
	}
	if (countBytes) out.byteCount += 32;
	if (countChars) out.charCount += popcnt32(maskUtf8Lead32(v));
}
inline void processTail(const unsigned char* buf, size_t n, Counts& out, Avx2State& st,
	bool countLines, bool countWords, bool countBytes,
	bool countChars, bool countMaxLine)
{
	for (size_t i = 0; i < n; ++i) {
		unsigned char c = buf[i];
		if (countBytes) out.byteCount++;
		if (countLines && c == '\n') out.lineCount++;
		if (countWords) {
			bool space = isSpaceAscii(c);
			uint32_t prev = st.prevSpaceBit;
			if (!space && prev) out.wordCount++;
			st.prevSpaceBit = space ? 1u : 0u;
		}
		if (countChars) if (isUtf8Lead(c)) out.charCount++;
	}
}
#endif

static void printCounts(const Counts& c, const std::string* label,
	bool lines, bool words, bool bytes, bool chars, bool maxLine)
{
	if (lines)   std::cout << c.lineCount << " ";
	if (words)   std::cout << c.wordCount << " ";
	if (bytes)   std::cout << c.byteCount << " ";
	if (chars)   std::cout << c.charCount << " ";
	if (maxLine) std::cout << c.maxLineLength << " ";
	if (label)   std::cout << *label;
	std::cout << "\n";
}

int main(int argc, char** argv) {
	initSpaceTable();
	Options opt;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a.size() > 1 && a[0] == '-' && a[1] != '-') {
			for (size_t j = 1; j < a.size(); ++j) {
				char ch = a[j];
				if (ch == 'l') opt.optLines = true;
				else if (ch == 'w') opt.optWords = true;
				else if (ch == 'c') opt.optBytes = true;
				else if (ch == 'm') opt.optChars = true;
				else if (ch == 'L') opt.optMaxLine = true;
			}
		}
		else {
			opt.files.push_back(a);
		}
	}
	if (!opt.optLines && !opt.optWords && !opt.optBytes && !opt.optChars && !opt.optMaxLine)
		opt.optLines = opt.optWords = opt.optBytes = true;
	if (opt.files.empty()) opt.files.push_back("-");

	std::vector<unsigned char> buffer(kBufSize);
	Counts total{};
	bool haveTotal = (opt.files.size() > 1);

	for (const auto& path : opt.files) {
		FILE* f = nullptr;
#ifdef _MSC_VER
		if (path == "-") {
			f = stdin;
		}
		else {
			errno_t err = fopen_s(&f, path.c_str(), "rb");
			if (err != 0 || !f) {
				std::cerr << "fastawc: cannot open " << path << "\n";
				continue;
			}
		}
#else
		if (path == "-") {
			f = stdin;
		}
		else {
			f = fopen(path.c_str(), "rb");
			if (!f) {
				std::cerr << "wc_fast: cannot open " << path << "\n";
				continue;
			}
		}
#endif

		Counts c{};
#ifndef __AVX2__
		ScalarState st{};
		for (;;) {
			size_t n = fread(buffer.data(), 1, buffer.size(), f);
			if (n == 0) break;
			processScalar(buffer.data(), n, c, st,
				opt.optLines, opt.optWords, opt.optBytes,
				opt.optChars, opt.optMaxLine);
		}
		finalizeScalar(c, st, opt.optMaxLine);
#else
		Avx2State st{};
		for (;;) {
			size_t n = fread(buffer.data(), 1, buffer.size(), f);
			if (n == 0) break;
			size_t i = 0;
			while (i + 32 <= n) {
				__m256i v = _mm256_loadu_si256((const __m256i*)(buffer.data() + i));
				processBlock32(v, c, st,
					opt.optLines, opt.optWords, opt.optBytes,
					opt.optChars, opt.optMaxLine);
				i += 32;
			}
			if (i < n) {
				processTail(buffer.data() + i, n - i, c, st,
					opt.optLines, opt.optWords, opt.optBytes,
					opt.optChars, opt.optMaxLine);
			}
		}
		if (opt.optMaxLine && st.currentLineLen > c.maxLineLength)
			c.maxLineLength = st.currentLineLen;
#endif

		if (path == "-") printCounts(c, nullptr,
			opt.optLines, opt.optWords, opt.optBytes,
			opt.optChars, opt.optMaxLine);
		else             printCounts(c, &path,
			opt.optLines, opt.optWords, opt.optBytes,
			opt.optChars, opt.optMaxLine);

		total.lineCount += c.lineCount;
		total.wordCount += c.wordCount;
		total.byteCount += c.byteCount;
		total.charCount += c.charCount;
		total.maxLineLength = std::max(total.maxLineLength, c.maxLineLength);

		if (path != "-") fclose(f);
	}

	if (haveTotal) {
		std::string label = "total";
		printCounts(total, &label,
			opt.optLines, opt.optWords, opt.optBytes,
			opt.optChars, opt.optMaxLine);
	}
	return 0;
}
