#[[
- Try to find Cairo
Once done, this will define

  CAIRO_FOUND - system has Cairo
  CAIRO_INCLUDE_DIRS - the Cairo include directories
  CAIRO_LIBRARIES - link these to use Cairo

Copyright (C) 2012 Raphael Kubo da Costa <rakuco@webkit.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
2.  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND ITS CONTRIBUTORS ``AS
IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR ITS
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#]]

find_package(PkgConfig QUIET)
pkg_check_modules(PC_CAIRO cairo QUIET) # FIXME: After we require CMake 2.8.2 we
                                        # can pass QUIET to this call.

find_path(
  CAIRO_INCLUDE_DIRS
  NAMES cairo.h
  HINTS ${PC_CAIRO_INCLUDEDIR} ${PC_CAIRO_INCLUDE_DIRS}
  PATH_SUFFIXES cairo)

find_library(
  CAIRO_LIBRARY_RELEASE
  NAMES cairo
  HINTS ${PC_CAIRO_LIBDIR} ${PC_CAIRO_LIBRARY_DIRS})

find_library(
  CAIRO_LIBRARY_DEBUG
  NAMES cairod
  HINTS ${PC_CAIRO_LIBDIR} ${PC_CAIRO_LIBRARY_DIRS})

set(CAIRO_LIBRARY)
if(CAIRO_LIBRARY_DEBUG)
  set(CAIRO_LIBRARY ${CAIRO_LIBRARY_DEBUG})
elseif(CAIRO_LIBRARY_RELEASE)
  set(CAIRO_LIBRARY ${CAIRO_LIBRARY_RELEASE})
endif()

if(CAIRO_INCLUDE_DIRS)
  if(EXISTS "${CAIRO_INCLUDE_DIRS}/cairo-version.h")
    file(READ "${CAIRO_INCLUDE_DIRS}/cairo-version.h" CAIRO_VERSION_CONTENT)

    string(REGEX MATCH "#define +CAIRO_VERSION_MAJOR +([0-9]+)" _dummy
                 "${CAIRO_VERSION_CONTENT}")
    set(CAIRO_VERSION_MAJOR "${CMAKE_MATCH_1}")

    string(REGEX MATCH "#define +CAIRO_VERSION_MINOR +([0-9]+)" _dummy
                 "${CAIRO_VERSION_CONTENT}")
    set(CAIRO_VERSION_MINOR "${CMAKE_MATCH_1}")

    string(REGEX MATCH "#define +CAIRO_VERSION_MICRO +([0-9]+)" _dummy
                 "${CAIRO_VERSION_CONTENT}")
    set(CAIRO_VERSION_MICRO "${CMAKE_MATCH_1}")

    set(CAIRO_VERSION
        "${CAIRO_VERSION_MAJOR}.${CAIRO_VERSION_MINOR}.${CAIRO_VERSION_MICRO}")
  endif()
endif()

# FIXME: Should not be needed anymore once we start depending on CMake 2.8.3
set(VERSION_OK TRUE)
if(Cairo_FIND_VERSION)
  if(Cairo_FIND_VERSION_EXACT)
    if("${Cairo_FIND_VERSION}" VERSION_EQUAL "${CAIRO_VERSION}")
      # FIXME: Use IF (NOT ...) with CMake 2.8.2+ to get rid of the ELSE block
    else()
      set(VERSION_OK FALSE)
    endif()
  else()
    if("${Cairo_FIND_VERSION}" VERSION_GREATER "${CAIRO_VERSION}")
      set(VERSION_OK FALSE)
    endif()
  endif()
endif()

find_path(FONTCONFIG_INCLUDE_DIR fontconfig/fontconfig.h)

if(FONTCONFIG_INCLUDE_DIR)
  set(CAIRO_INCLUDE_DIRS ${CAIRO_INCLUDE_DIRS} ${FONTCONFIG_INCLUDE_DIR})
else()
  message(
    STATUS
      "fontconfig/fontconfig.h was not found. \n I had to unset(CAIRO_INCLUDE_DIRS) to make find_package() fail \n "
  )
  unset(CAIRO_INCLUDE_DIRS CACHE)
endif()

find_library(FONTCONFIG_LIBRARY NAMES fontconfig)
if(FONTCONFIG_LIBRARY)
  set(CAIRO_LIBRARIES ${CAIRO_LIBRARY} ${FONTCONFIG_LIBRARY})
else()
  message(
    STATUS
      "fontconfig library file was not found. \n I had to unset(CAIRO_LIBRARIES) to make find_package() fail \n "
  )
  unset(CAIRO_LIBRARIES CACHE)
endif()

mark_as_advanced(CAIRO_INCLUDE_DIRS CAIRO_LIBRARY CAIRO_LIBRARY_RELEASE
                 CAIRO_LIBRARY_DEBUG FONTCONFIG_LIBRARY FONTCONFIG_INCLUDE_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cairo DEFAULT_MSG CAIRO_INCLUDE_DIRS
                                  CAIRO_LIBRARIES VERSION_OK)
