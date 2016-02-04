<?php

function render_table($d)
{
	echo "<p>";
	echo "<table border=1 cellpadding=3>";
	echo "<tr><th>pred</th>";
	for ($i=1; $i<33; $i++)
		echo "<th>$i</th>";
	echo "</tr>";
	foreach ($d as $pred => $data)
	{
		echo "<tr><td>$pred</td>";
		for ($i=1; $i<33; $i++)
		{
			echo "<td><img src=\"images/".$data[$i][0].".JPG\" style=\"width:100px;height:100px;\">";
			echo "<br/>".$data[$i][1]."</td>";
		}
	}
	echo "</table>";
	echo "</p>";
}
function read_data_from_file($cond)
{
	$input_fn = $cond.'_xval_preds_results.txt';
	$f = fopen($input_fn, 'r');
	$d = array();
	if (!$f)
		return $d;
	while (!feof($f))
	{
		$l = trim(fgets($f));
		if (strlen($l) > 0)
		{
			$pred_data = explode(":", $l);
			$pred = $pred_data[0];
			$obj_decs = explode(";", $pred_data[1]);
			$pred_decs = array();
			for ($i=0; $i<count($obj_decs); $i++)
			{
				$obj_dec = explode(",", $obj_decs[$i]);
				array_push($pred_decs, $obj_dec);
			}
			$d[$pred] = $pred_decs;
		}
	}
	fclose($f);
	return $d;
}

?>

<?php

$cond = $_GET['cond'];

$d = read_data_from_file($cond);
render_table($d);

?>
