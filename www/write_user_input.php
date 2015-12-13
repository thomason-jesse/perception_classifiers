<?php

if (!isset($_POST))
{
	die("FAILED<br/>no POST detected");
}
if (!isset($_POST['user_id']))
{
	die("FAILED<br/>no user_id posted");
}
if (!isset($_POST['user_input']) && !isset($_POST['user_guess']))
{
	die("FAILED<br/>no user_input or user_guess posted");
}

$path_to_dialog = '';
$user_id = $_POST['user_id'];

// write user input, if any, to file to be read by agent to generate a response
if (isset($_POST['user_input']))
{
	$user_input_raw = $_POST['user_input'];
	$user_input = htmlspecialchars($user_input_raw);
	$input_file = fopen($path_to_dialog.'communications/'.$user_id.'.get.in', 'w');
	fwrite($input_file, $user_input);
	fclose($input_file);
	exec("chmod 777 ".$input_file);
}

// write user guess, if any, to file to be read by agent to generate a response
if (isset($_POST['user_guess']))
{
	$user_guess = $_POST['user_guess'];
	$input_file = fopen($path_to_dialog.'communications/'.$user_id.'.guess.in', 'w');
	fwrite($input_file, $user_guess);
	fclose($input_file);
	exec("chmod 777 ".$input_file);
}

?>
