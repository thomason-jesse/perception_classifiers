<?php

function render_password_form()
{
	echo "<p><form method=\"post\">enter password:<br/>";
	echo "<input type=\"password\" name=\"pw\">";
	echo "<br/><input type=\"submit\" value=\"submit\"></form>";
	echo "</p>";
}
function render_table()
{
	echo "<p><form method=\"post\">";
	echo "<input type=\"submit\" value=\"Add\">";
	echo "<input type=\"hidden\" name=\"pw\" value=\"".($_POST['pw'])."\">";
	echo "<table border=1 cellpadding=3>";
	echo "<tr><th>ID</th><th>Name</th><th>Fold</th><th>Object IDs</th><th>Object Pics</th><th>Command</th></tr>";
	echo "<tr><td>&nbsp;</td>";
	echo "<td><input type=\"text\" name=\"user_name\"></td>";
	echo "<td><input type=\"text\" name=\"fold\"></td>";
	echo "<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td></tr>";
	write_table_rows_from_file();
	echo "</table>";
	echo "</form></p>";
}
function get_object_ids($fold)
{
	shuffle($fold);
	$fold_a = array();
	$fold_b = array();
	for ($i=0; $i<count($fold); $i++)
	{
		if ($i%2 == 0)
			array_push($fold_a, $fold[$i]);
		else
			array_push($fold_b, $fold[$i]);
	}
	$a_str = implode(', ', $fold_a);
	$b_str = implode(', ', $fold_b);
	return implode(';', array($a_str, $b_str));
}
function write_table_rows_from_file()
{
	$entries = read_data_from_file();
	for ($i=count($entries)-1; $i>=0; $i--)
	{
		echo "<tr>";
		for ($j=0; $j<count($entries[$i]); $j++)
		{
			echo "<td>".$entries[$i][$j]."</td>";
		}
		$image_array = explode(', ', $entries[$i][count($entries[$i])-1]);
		echo "<td>".image_list($image_array)."</td>";
		echo "<td>".build_command($entries[$i][0], $image_array)."</td>";
		echo "</tr>";
	}
}
function image_list($l)
{
	$s = "";
	for ($i=0; $i<count($l); $i++)
	{
		$s .= "<img src=\"images/".$l[$i].".JPG\" style=\"width:100px;height:100px;\">";
	}
	return $s;
}
function build_command($id, $l)
{
	$s = "rosrun perception_classifiers ispy.py ";
	$s .= implode(',', $l);
	$s .= " 2 ".$id." robot [agent_to_load]";
	return $s;
}
function read_data_from_file()
{
	$input_fn = 'robot_users_data.txt';
	$f = fopen($input_fn, 'r');
	$entries = array();
	if (!$f)
		return $entries;
	while (!feof($f))
	{
		$l = trim(fgets($f));
		if (strlen($l) > 0)
		{
			$d = explode("|", $l);
			array_push($entries, $d);
		}
	}
	fclose($f);
	return $entries;
}
function next_available_id()
{
	$entries = read_data_from_file();
	$max_id = -1;
	for ($i=0; $i<count($entries); $i++)
	{
		if ($entries[$i][0] > $max_id)
		{
			$max_id = $entries[$i][0];
		}
	}
	return $max_id+1;
}

?>

<?php

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
if (!isset($_POST['user_name']) || !isset($_POST['fold']))
{
	render_table();
	die();
}

$user_name = $_POST['user_name'];
$user_fold = $_POST['fold'];

// calculate next available id
$user_id = next_available_id();

// get a randomized object ordering for user
$folds = array();
array_push($folds, array(29, 3, 26, 19, 16, 27, 5, 32));
array_push($folds, array(30, 12, 11, 14, 6, 25, 7, 8));
array_push($folds, array(31, 9, 4, 23, 2, 13, 22, 24));
array_push($folds, array(18, 17, 28, 1, 15, 10, 21, 20));
$object_ids_str = get_object_ids($folds[$user_fold]);
$object_ids_folds = explode(";", $object_ids_str);
$fold_letters = array('a', 'b');

// write new information to file
$input_fn = 'robot_users_data.txt';
$input_file = fopen($input_fn, 'a') or die("<p><div style=\"color:red\">unable to open data file</div></p>");
for ($i=1; $i>=0; $i--)
{
	$data = implode("|",array($user_id+$i, $user_name, $user_fold.$fold_letters[$i], $object_ids_folds[$i]));
	fwrite($input_file, $data."\n");
}
fclose($input_file);
render_table();

?>
