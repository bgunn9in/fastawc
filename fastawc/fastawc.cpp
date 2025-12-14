#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

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
	bool useAvx2 = false;
	std::optional<std::string> filesFrom;
	std::vector<std::string> files;
};

static constexpr size_t bufSize = 1 << 20;

inline bool isSpaceAscii(unsigned char c) {
	switch (c) {
	case ' ': case '\n': case '\t': case '\r': case '\v': case '\f': return true;
	default: return false;
	}
}

inline bool isUtf8Lead(unsigned char c) {
	return (c & 0xC0) != 0x80;
}

struct ScalarState {
	bool prevSpace = true;
	uint64_t currentLineLength = 0;
};

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
				if (countMaxLine) st.currentLineLength++;
			}
		}
		else if (countMaxLine) {
			st.currentLineLength++;
		}
		if (countMaxLine && c == '\n') {
			if (st.currentLineLength > out.maxLineLength) out.maxLineLength = st.currentLineLength;
			st.currentLineLength = 0;
		}
	}
}

inline void finalizeScalar(Counts& out, ScalarState& st, bool countMaxLine) {
	if (countMaxLine) {
		if (st.currentLineLength > out.maxLineLength) out.maxLineLength = st.currentLineLength;
	}
}

#ifdef __AVX2__
struct Avx2State {
	uint8_t prevSpace = 1;
	uint64_t currentLineLength = 0;
};

inline uint32_t popcnt32(uint32_t x) {
	return (uint32_t)_mm_popcnt_u32(x);
}

inline uint32_t eqMask32(__m256i v, unsigned char ch) {
	__m256i m = _mm256_cmpeq_epi8(v, _mm256_set1_epi8((char)ch));
	return (uint32_t)_mm256_movemask_epi8(m);
}

inline uint32_t whitespaceMask32(__m256i v) {
	uint32_t m =
		eqMask32(v, ' ') |
		eqMask32(v, '\n') |
		eqMask32(v, '\t') |
		eqMask32(v, '\r') |
		eqMask32(v, '\v') |
		eqMask32(v, '\f');
	return m;
}

inline void processAvx2(const unsigned char* buf, size_t n, Counts& out, Avx2State& st,
	bool countLines, bool countWords, bool countBytes,
	bool countChars, bool countMaxLine)
{
	if (countBytes) out.byteCount += n;
	size_t i = 0;
	while (i + 32 <= n) {
		__m256i v = _mm256_loadu_si256((const __m256i*)(buf + i));
		if (countLines) {
			uint32_t nlm = eqMask32(v, '\n');
			out.lineCount += popcnt32(nlm);
		}
		if (countWords) {
			uint32_t wsm = whitespaceMask32(v);
			uint32_t prev = (wsm << 1) | (st.prevSpace ? 1u : 0u);
			uint32_t startMask = (~wsm) & prev;
			out.wordCount += popcnt32(startMask);
			st.prevSpace = (wsm >> 31) & 1u;
		}
		for (int k = 0; k < 32; ++k) {
			unsigned char c = buf[i + k];
			if (countChars && isUtf8Lead(c)) {
				out.charCount++;
				if (countMaxLine) st.currentLineLength++;
			}
			else if (countMaxLine && !countChars) {
				st.currentLineLength++;
			}
			if (countMaxLine && c == '\n') {
				if (st.currentLineLength > out.maxLineLength) out.maxLineLength = st.currentLineLength;
				st.currentLineLength = 0;
			}
		}
		i += 32;
	}
	for (; i < n; ++i) {
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
				if (countMaxLine) st.currentLineLength++;
			}
		}
		else if (countMaxLine) {
			st.currentLineLength++;
		}
		if (countMaxLine && c == '\n') {
			if (st.currentLineLength > out.maxLineLength) out.maxLineLength = st.currentLineLength;
			st.currentLineLength = 0;
		}
	}
}

inline void finalizeAvx2(Counts& out, Avx2State& st, bool countMaxLine) {
	if (countMaxLine) {
		if (st.currentLineLength > out.maxLineLength) out.maxLineLength = st.currentLineLength;
	}
}
#endif

struct Reader {
	bool isStdin = false;
#ifdef _WIN32
	HANDLE hFile = INVALID_HANDLE_VALUE;
#else
	int fd = -1;
#endif
	std::vector<unsigned char> buffer;
	Reader() : buffer(bufSize) {}
	bool openStdin() {
		isStdin = true;
#ifdef _WIN32
		_setmode(_fileno(stdin), _O_BINARY);
#endif
		return true;
	}
	bool openFile(const std::string& path, std::string& err) {
		isStdin = false;
#ifdef _WIN32
		hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
			nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (hFile == INVALID_HANDLE_VALUE) {
			err = "cannot open '" + path + "'";
			return false;
		}
#else
		fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
		if (fd < 0) {
			err = "cannot open '" + path + "'";
			return false;
		}
#endif
		return true;
	}
	std::optional<size_t> readChunk() {
#ifdef _WIN32
		if (isStdin) {
			size_t n = fread(buffer.data(), 1, buffer.size(), stdin);
			if (n == 0) {
				if (feof(stdin)) return size_t(0);
				return std::optional<size_t>{};
			}
			return n;
		}
		else {
			DWORD readBytes = 0;
			if (!ReadFile(hFile, buffer.data(), (DWORD)buffer.size(), &readBytes, nullptr)) {
				return std::optional<size_t>{};
			}
			return (size_t)readBytes;
		}
#else
		int fdUse = isStdin ? STDIN_FILENO : fd;
		ssize_t n = ::read(fdUse, buffer.data(), buffer.size());
		if (n < 0) return std::optional<size_t>{};
		return (size_t)n;
#endif
	}
	void close() {
#ifdef _WIN32
		if (!isStdin && hFile != INVALID_HANDLE_VALUE) {
			CloseHandle(hFile);
			hFile = INVALID_HANDLE_VALUE;
		}
#else
		if (!isStdin && fd >= 0) {
			::close(fd);
			fd = -1;
		}
#endif
	}
};

bool readFiles0From(const std::string& spec, std::vector<std::string>& out, std::string& err) {
	Reader r;
	if (spec == "-") {
		r.openStdin();
	}
	else {
		if (!r.openFile(spec, err)) return false;
	}
	std::string acc;
	for (;;) {
		auto nopt = r.readChunk();
		if (!nopt.has_value()) { err = "read error"; r.close(); return false; }
		size_t n = *nopt;
		if (n == 0) break;
		const unsigned char* p = r.buffer.data();
		for (size_t i = 0; i < n; ++i) {
			if (p[i] == '\0') {
				out.push_back(std::move(acc));
				acc.clear();
			}
			else {
				acc.push_back((char)p[i]);
			}
		}
	}
	if (!acc.empty()) out.push_back(std::move(acc));
	r.close();
	return true;
}

struct DisplayFlags {
	bool lines = false;
	bool words = false;
	bool bytes = false;
	bool chars = false;
	bool maxLine = false;
};

std::string padRight(const std::string& s, size_t width) {
	if (s.size() >= width) return s;
	return std::string(width - s.size(), ' ') + s;
}

void printCounts(const Counts& c, const std::string* label, const DisplayFlags& d) {
	std::vector<std::string> cols;
	if (d.lines)   cols.push_back(std::to_string(c.lineCount));
	if (d.words)   cols.push_back(std::to_string(c.wordCount));
	if (d.bytes)   cols.push_back(std::to_string(c.byteCount));
	if (d.chars)   cols.push_back(std::to_string(c.charCount));
	if (d.maxLine) cols.push_back(std::to_string(c.maxLineLength));
	for (size_t i = 0; i < cols.size(); ++i) {
		cols[i] = padRight(cols[i], 7);
	}
	for (const auto& s : cols) {
		std::fwrite(s.data(), 1, s.size(), stdout);
		std::fwrite(" ", 1, 1, stdout);
	}
	if (label) {
		std::fwrite(label->c_str(), 1, label->size(), stdout);
	}
	std::fwrite("\n", 1, 1, stdout);
}

void printHelp() {
	std::cout <<
		R"(Usage: fastawc [OPTION]... [FILE]...
Print newline, word, and byte counts for each FILE, and a total line if more than one FILE is specified.
With no FILE, or when FILE is -, read standard input.

  -c, --bytes              print the byte counts
  -m, --chars              print the character counts (UTF-8 code points)
  -l, --lines              print the newline counts
  -w, --words              print the word counts (ASCII whitespace)
  -L, --max-line-length    print the maximum display width (UTF-8 code points)
	  --files0-from=FILE   read input file names from FILE, NUL-separated; '-' for stdin
	  -avx2                use AVX2 engine (if compiled with AVX2)
	  --help               display this help and exit
	  --version            output version information and exit

By default, fastawc prints line, word, and byte counts.
Note: character and max-line-length counts use UTF-8 code points; locale-dependent width is not computed.)";
}

void printVersion() {
	std::cout << "fastawc 1.0 (C++20, scalar+AVX2)\n";
}

bool parseOptions(int argc, char** argv, Options& opt, std::string& err) {
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a == "--help") { printHelp(); std::exit(0); }
		else if (a == "--version") { printVersion(); std::exit(0); }
		else if (a == "-c" || a == "--bytes") opt.optBytes = true;
		else if (a == "-m" || a == "--chars") opt.optChars = true;
		else if (a == "-l" || a == "--lines") opt.optLines = true;
		else if (a == "-w" || a == "--words") opt.optWords = true;
		else if (a == "-L" || a == "--max-line-length") opt.optMaxLine = true;
		else if (a.rfind("--files0-from=", 0) == 0) {
			opt.filesFrom = a.substr(std::string("--files0-from=").size());
		}
		else if (a == "-avx2") {
#ifdef __AVX2__
			opt.useAvx2 = true;
#else
			err = "binary not built with AVX2 support";
			return false;
#endif
		}
		else if (!a.empty() && a[0] == '-') {
			if (a == "-") opt.files.push_back("-");
			else { err = "invalid option: " + a; return false; }
		}
		else {
			opt.files.push_back(a);
		}
	}
	if (!opt.optLines && !opt.optWords && !opt.optBytes && !opt.optChars && !opt.optMaxLine) {
		opt.optLines = opt.optWords = opt.optBytes = true;
	}
	if (opt.filesFrom.has_value()) {
		std::vector<std::string> loaded;
		if (!readFiles0From(*opt.filesFrom, loaded, err)) return false;
		opt.files.insert(opt.files.end(), loaded.begin(), loaded.end());
	}
	if (opt.files.empty()) {
		opt.files.push_back("-");
	}
	return true;
}

int main(int argc, char** argv) {
	std::ios::sync_with_stdio(false);
	Options opt;
	std::string err;
	if (!parseOptions(argc, argv, opt, err)) {
		if (!err.empty()) {
			std::fprintf(stderr, "fastawc: %s\n", err.c_str());
		}
		return 1;
	}
	DisplayFlags disp{
		.lines = opt.optLines,
		.words = opt.optWords,
		.bytes = opt.optBytes,
		.chars = opt.optChars,
		.maxLine = opt.optMaxLine
	};
	Counts total{};
	bool haveTotal = (opt.files.size() > 1);
	for (size_t idx = 0; idx < opt.files.size(); ++idx) {
		const std::string& path = opt.files[idx];
		Reader r;
		if (path == "-") {
			r.openStdin();
		}
		else {
			if (!r.openFile(path, err)) {
				std::fprintf(stderr, "fastawc: %s\n", err.c_str());
				continue;
			}
		}
		Counts c{};
#ifdef __AVX2__
		if (opt.useAvx2) {
			Avx2State st{};
			for (;;) {
				auto nopt = r.readChunk();
				if (!nopt.has_value()) { std::fprintf(stderr, "fastawc: read error\n"); break; }
				size_t n = *nopt;
				if (n == 0) break;
				processAvx2(r.buffer.data(), n, c, st,
					disp.lines, disp.words, disp.bytes, disp.chars, disp.maxLine);
			}
			finalizeAvx2(c, st, disp.maxLine);
		}
		else
#endif
		{
			ScalarState st{};
			for (;;) {
				auto nopt = r.readChunk();
				if (!nopt.has_value()) { std::fprintf(stderr, "fastawc: read error\n"); break; }
				size_t n = *nopt;
				if (n == 0) break;
				processScalar(r.buffer.data(), n, c, st,
					disp.lines, disp.words, disp.bytes, disp.chars, disp.maxLine);
			}
			finalizeScalar(c, st, disp.maxLine);
		}
		r.close();
		if (path == "-") {
			printCounts(c, nullptr, disp);
		}
		else {
			printCounts(c, &path, disp);
		}
		total.lineCount += c.lineCount;
		total.wordCount += c.wordCount;
		total.byteCount += c.byteCount;
		total.charCount += c.charCount;
		total.maxLineLength = std::max(total.maxLineLength, c.maxLineLength);
	}
	if (haveTotal) {
		std::string totalLabel = "total";
		printCounts(total, &totalLabel, disp);
	}
	return 0;
}
