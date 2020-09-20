QT += core gui widgets

CONFIG += qt c++11


SOURCES += \
    ../../../src/ColorSpaces.cpp \
    ../../../src/ImageFilter.cpp \
    ../../../src/ImageProcessing.cpp \
    ../../../src/main.cpp \
    ../../../src/NoiseReduction.cpp
	

HEADERS += \
    ../../../inc/ColorSpaces.h \
    ../../../inc/ImageProcessing.h \
	../../../inc/ImageFilter.h \
	../../../inc/NoiseReduction.h


INCLUDEPATH += ../../../lib_inc/ ../../../inc/


unix:!macx|win32: LIBS += -L$$PWD/../../../lib/windows-mingw/ -lImageDSPLib

win32:!win32-g++: PRE_TARGETDEPS += $$PWD/../../../lib/windows-mingw/ImageDSPLib.lib


