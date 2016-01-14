<?php

function print_randomized_fold($fold)
{
	shuffle($fold);
	echo "<p>";
	for ($i=0; $i<8; $i++)
	{
		echo $fold[$i].", ";
	}
	echo "</p>";
}

function render_fold_form()
{
	echo "<p><form method=\"post\">";
	echo "<table>";
	echo "<tr><td>Fold:</td>";
	echo "<td><input type=\"text\" name=\"fold\"></td></tr>";
	echo "<tr><td><input type=\"submit\" value=\"submit\"></form></td>";
	echo "<td>&nbsp;</td></tr>";
	echo "</table>";
	echo "</form></p>";
}

if (!isset($_POST) || !isset($_POST['fold']))
{
	render_fold_form();
	die();
}

$folds = array();
array_push($folds, array(29, 3, 26, 19, 16, 27, 5, 32));
array_push($folds, array(30, 12, 11, 14, 6, 25, 7, 8));
array_push($folds, array(31, 9, 4, 23, 2, 13, 22, 24));
array_push($folds, array(18, 17, 28, 1, 15, 10, 21, 20));

echo "fold ".$_POST['fold']."<br/>";
print_randomized_fold($folds[$_POST['fold']]);

render_fold_form();

?>
