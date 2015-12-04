cmake_minimum_required(VERSION 2.8.3)
project(perception_classifiers)

find_package(OpenCV)

find_package(catkin REQUIRED 
    COMPONENTS
        roscpp
        std_msgs
        bwi_kr_execution
        cv_bridge
)


find_package(Boost REQUIRED COMPONENTS filesystem)

add_service_files(
	FILES
    getFreeClassifierID.srv
		loadClassifiers.srv
		runClassifier.srv
		trainClassifier.srv
    startDialog.srv
    getSay.srv
    getPoint.srv)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
    CATKIN_DEPENDS 
        roscpp
        std_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

add_executable(classifier_services src/classifier_services.cpp)
target_link_libraries(classifier_services
		${catkin_LIBRARIES}
		${OpenCV_LIBRARIES})

add_dependencies(classifier_services perception_classifiers)

#############
## Install ##
#############

# install(TARGETS perception_classifiers
#     RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

