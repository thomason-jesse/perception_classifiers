<?php

function render_password_form()
{
	echo "<p><form method=\"post\">enter password:<br/>";
	echo "<input type=\"password\" name=\"pw\">";
	echo "<br/><input type=\"submit\" value=\"submit\"></form>";
	echo "</p>";
}

function render_scribe_form()
{
	echo "<p><form method=\"post\">";
	echo "<input type=\"hidden\" name=\"pw\" value=\"".($_POST['pw'])."\">";
	echo "<table>";
	echo "<tr><td>User ID:</td>";
	echo "<td><input type=\"text\" name=\"user_id\"></td></tr>";
	echo "<tr><td>User Speech:</td>";
	echo "<td><input type=\"text\" name=\"user_input\"></td></tr>";
	echo "<tr><td><input type=\"submit\" value=\"submit\"></form></td>";
	echo "<td>&nbsp;</td></tr>";
	echo "</table>";
	echo "</form></p>";
}

if (!isset($_POST) || !isset($_POST['pw']))
{
	render_password_form();
	die();
}
if (strcmp($_POST['pw'],'frozenfish3768') != 0)
{
	echo "<p><div style=\"color:red\">";
	echo "password incorrect";
	echo "</div></p>";
	render_password_form();
	die();
}
if (!isset($_POST['user_id']) || !isset($_POST['user_input']))
{
	render_scribe_form();
	die();
}

$path_to_dialog = '';
$user_id = $_POST['user_id'];

// write user input, if any, to file to be read by agent to generate a response
$user_input_raw = $_POST['user_input'];
$user_input = htmlspecialchars($user_input_raw);
$input_fn = $path_to_dialog.'communications/'.$user_id.'.get.in';
$input_file = fopen($input_fn, 'w');
fwrite($input_file, $user_input);
fclose($input_file);
exec("chmod 777 ".$input_file);

echo "<p>Wrote '".$user_input_raw."' to file ".$input_fn."</p>";
render_scribe_form();

?>
