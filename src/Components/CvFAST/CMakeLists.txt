# Include the directory itself as a path to include directories
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a variable containing all .cpp files:
FILE(GLOB files *.cpp)

# Find OpenCV library files
FIND_PACKAGE( OpenCV REQUIRED )

IF (${OpenCV_VERSION} VERSION_GREATER 2.3.9)

# Create an executable file from sources:
ADD_LIBRARY(CvFAST SHARED ${files})

# Link external libraries
TARGET_LINK_LIBRARIES(CvFAST ${DisCODe_LIBRARIES} ${OpenCV_LIBS})

INSTALL_COMPONENT(CvFAST)

ELSE ()

MESSAGE ( STATUS "Skipping FAST - required OpenCV 2.4. Detected OpenCV ${OpenCV_VERSION}.")

ENDIF ()
