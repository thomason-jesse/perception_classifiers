<?php

if (!isset($_POST))
{
	die("FAILED\nno POST detected");
}
if (!isset($_POST['user_id']))
{
	die("FAILED\nno user_id posted");
}

$user_id = $_POST['user_id'];

//write MTurk validation to output
$mturk_code = $user_id."_".substr(sha1("ispy_salted_hash_is_frozenfish".$user_id),0,13);
echo "<p>Thank you for your participation!</p><p>Copy the code below, return to Mechanical Turk, and enter it to receive payment:<br/>".$mturk_code."</p>";

?>