#include "platform.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fastawc {

namespace {

void reset_source(FileSource& source) noexcept {
	source.kind = FileSource::Kind::none;
	source.data = nullptr;
	source.size = 0;
	source.stream = nullptr;
	source.closeStream = false;
#ifdef _WIN32
	source.mapping = nullptr;
#else
	source.fd = -1;
	source.closeFd = false;
#endif
}

void close_source(FileSource& source) noexcept {
	if (source.kind == FileSource::Kind::mapped) {
#ifdef _WIN32
		if (source.data != nullptr) {
			UnmapViewOfFile(source.data);
		}
		if (source.mapping != nullptr) {
			CloseHandle(source.mapping);
		}
#else
		if (source.data != nullptr && source.size != 0) {
			munmap(const_cast<uint8_t*>(source.data), source.size);
		}
#endif
	}

#ifndef _WIN32
	if (source.closeFd && source.fd >= 0) {
		close(source.fd);
	}
#endif

	if (source.closeStream && source.stream != nullptr) {
		fclose(source.stream);
	}

	reset_source(source);
}

void prepare_stream(FILE* const file) noexcept {
	setvbuf(file, nullptr, _IOFBF, 1 << 20);
#ifndef _WIN32
#if defined(POSIX_FADV_SEQUENTIAL)
	const int fd = fileno(file);
	if (fd >= 0) {
		posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
	}
#endif
#endif
}

bool env_flag_enabled(const char* const name) noexcept {
	const char* const raw = std::getenv(name);
	if (raw == nullptr || *raw == '\0') {
		return false;
	}

	const std::string_view value{ raw };
	return value != "0" && value != "false" && value != "FALSE" && value != "False";
}

#ifdef _WIN32
bool try_map_file_windows(const std::string& path, FileSource& source) noexcept {
	HANDLE file = CreateFileA(
		path.c_str(),
		GENERIC_READ,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
		nullptr,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
		nullptr);
	if (file == INVALID_HANDLE_VALUE) {
		return false;
	}

	LARGE_INTEGER fileSize{};
	if (!GetFileSizeEx(file, &fileSize) || fileSize.QuadPart < 0) {
		CloseHandle(file);
		return false;
	}

	source.size = static_cast<size_t>(fileSize.QuadPart);
	if (source.size == 0) {
		source.kind = FileSource::Kind::mapped;
		CloseHandle(file);
		return true;
	}

	source.mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
	CloseHandle(file);
	if (source.mapping == nullptr) {
		reset_source(source);
		return false;
	}

	source.data = static_cast<const uint8_t*>(MapViewOfFile(source.mapping, FILE_MAP_READ, 0, 0, 0));
	if (source.data == nullptr) {
		CloseHandle(source.mapping);
		reset_source(source);
		return false;
	}

	if (env_flag_enabled("FASTAWC_WILLNEED")) {
		WIN32_MEMORY_RANGE_ENTRY range{};
		range.VirtualAddress = const_cast<uint8_t*>(source.data);
		range.NumberOfBytes = source.size;
		(void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
	}

	source.kind = FileSource::Kind::mapped;
	return true;
}
#else
bool try_map_file_posix(const std::string& path, FileSource& source) noexcept {
	source.fd = open(path.c_str(), O_RDONLY);
	if (source.fd < 0) {
		return false;
	}
	source.closeFd = true;

	struct stat st {};
	if (fstat(source.fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) {
		close(source.fd);
		source.fd = -1;
		source.closeFd = false;
		return false;
	}

	source.size = static_cast<size_t>(st.st_size);
	if (source.size == 0) {
		source.kind = FileSource::Kind::mapped;
		return true;
	}

#if defined(POSIX_FADV_WILLNEED)
	if (env_flag_enabled("FASTAWC_WILLNEED")) {
		posix_fadvise(source.fd, 0, 0, POSIX_FADV_WILLNEED);
	}
#endif

	void* view = mmap(nullptr, source.size, PROT_READ, MAP_PRIVATE, source.fd, 0);
	if (view == MAP_FAILED) {
		close(source.fd);
		source.fd = -1;
		source.closeFd = false;
		reset_source(source);
		return false;
	}

#if defined(MADV_SEQUENTIAL)
	madvise(view, source.size, MADV_SEQUENTIAL);
#endif
#if defined(MADV_WILLNEED)
	if (env_flag_enabled("FASTAWC_WILLNEED")) {
		madvise(view, source.size, MADV_WILLNEED);
	}
#endif
	source.data = static_cast<const uint8_t*>(view);
	source.kind = FileSource::Kind::mapped;
	return true;
}
#endif

} // namespace

FileSource::FileSource(FileSource&& other) noexcept {
	*this = std::move(other);
}

FileSource& FileSource::operator=(FileSource&& other) noexcept {
	if (this != &other) {
		close_source(*this);
		kind = other.kind;
		data = other.data;
		size = other.size;
		stream = other.stream;
		closeStream = other.closeStream;
#ifdef _WIN32
		mapping = other.mapping;
		other.mapping = nullptr;
#else
		fd = other.fd;
		closeFd = other.closeFd;
		other.fd = -1;
		other.closeFd = false;
#endif
		other.kind = Kind::none;
		other.data = nullptr;
		other.size = 0;
		other.stream = nullptr;
		other.closeStream = false;
	}
	return *this;
}

FileSource::~FileSource() {
	close_source(*this);
}

bool open_regular_file(const std::string& path, FileSource& source, std::string& error) noexcept {
	close_source(source);

	const bool disableMapping = env_flag_enabled("FASTAWC_NO_MMAP");
#ifdef _WIN32
	if (!disableMapping && try_map_file_windows(path, source)) {
		return true;
	}
#else
	if (!disableMapping && try_map_file_posix(path, source)) {
		return true;
	}

	source.fd = open(path.c_str(), O_RDONLY);
	if (source.fd >= 0) {
#if defined(POSIX_FADV_SEQUENTIAL)
		posix_fadvise(source.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
#if defined(POSIX_FADV_WILLNEED)
		if (env_flag_enabled("FASTAWC_WILLNEED")) {
			posix_fadvise(source.fd, 0, 0, POSIX_FADV_WILLNEED);
		}
#endif
		source.kind = FileSource::Kind::stream;
		source.closeFd = true;
		return true;
	}
#endif

	FILE* const file = std::fopen(path.c_str(), "rb");
	if (file == nullptr) {
		error = std::strerror(errno);
		return false;
	}

	source.kind = FileSource::Kind::stream;
	source.stream = file;
	source.closeStream = true;
	prepare_stream(file);
	return true;
}

void open_stdin(FileSource& source) noexcept {
	close_source(source);
	source.kind = FileSource::Kind::stream;
#ifndef _WIN32
	source.fd = STDIN_FILENO;
	source.closeFd = false;
#endif
	source.stream = stdin;
#ifdef _WIN32
	_setmode(_fileno(stdin), _O_BINARY);
#endif
	prepare_stream(stdin);
}

size_t read_stream(FileSource& source, uint8_t* const buffer, const size_t bufferSize) noexcept {
#ifdef _WIN32
	return std::fread(buffer, 1, bufferSize, source.stream);
#else
	if (source.stream != nullptr) {
		return std::fread(buffer, 1, bufferSize, source.stream);
	}

	ssize_t readBytes = 0;
	do {
		readBytes = ::read(source.fd, buffer, bufferSize);
	} while (readBytes < 0 && errno == EINTR);
	return readBytes > 0 ? static_cast<size_t>(readBytes) : 0u;
#endif
}

} // namespace fastawc
