cmake_minimum_required(VERSION 2.8.3)
project(perception_classifiers)

find_package(OpenCV)

find_package(catkin REQUIRED 
    COMPONENTS
        roscpp
        std_msgs
        bwi_kr_execution
        cv_bridge
        message_generation
)


find_package(Boost REQUIRED COMPONENTS filesystem)

add_service_files(
	FILES
    getFreeClassifierID.srv
		loadClassifiers.srv
		runClassifier.srv
		trainClassifier.srv
		FetchFeatures.srv
    startDialog.srv
    getSay.srv
    getPoint.srv
)

add_message_files(
   FILES
   Observations.msg
)
 
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
add_executable(fetch_features src/fetch_features_service.cpp)

target_link_libraries(classifier_services
		${catkin_LIBRARIES}
		${OpenCV_LIBRARIES})

target_link_libraries(fetch_features ${catkin_LIBRARIES} ${BOOST_LIBRARIES})


add_dependencies(classifier_services perception_classifiers)
add_dependencies(fetch_features services_generate_messages_cpp)

#############
## Install ##
#############

# install(TARGETS perception_classifiers
#     RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

