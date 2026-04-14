#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace fastawc {

struct FileSource {
	enum class Kind {
		none,
		mapped,
		stream,
	};

	Kind kind = Kind::none;
	const uint8_t* data = nullptr;
	size_t size = 0;
	FILE* stream = nullptr;
	bool closeStream = false;

#ifdef _WIN32
	HANDLE mapping = nullptr;
#else
	int fd = -1;
	bool closeFd = false;
#endif

	FileSource() = default;
	FileSource(const FileSource&) = delete;
	FileSource& operator=(const FileSource&) = delete;
	FileSource(FileSource&& other) noexcept;
	FileSource& operator=(FileSource&& other) noexcept;
	~FileSource();

	bool is_mapped() const noexcept { return kind == Kind::mapped; }
	bool is_stream() const noexcept { return kind == Kind::stream; }
};

bool open_regular_file(const std::string& path, FileSource& source, std::string& error) noexcept;
void open_stdin(FileSource& source) noexcept;
size_t read_stream(FileSource& source, uint8_t* buffer, size_t bufferSize) noexcept;

} // namespace fastawc
