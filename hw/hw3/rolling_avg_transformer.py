from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class RollingAverageTransform(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(RollingAverageTransform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_col = self.getInputCols()
        output_col = self.getOutputCol()

        # Create a WindowSpec for 100 day window
        winSpec = (
            Window()
            .partitionBy(input_col[0])
            .orderBy("local_date_to_Unix")
            .rangeBetween(-99 * 86400, Window.currentRow)
        )

        dataset2 = (
            dataset
            # need to convert local_date to Unix Timestamp for Window time frame
            .withColumn(
                "local_date_to_Unix", F.unix_timestamp(input_col[3], "yyyy-MM-dd")
            ).withColumn(
                output_col,
                F.sum(input_col[1]).over(winSpec) / F.sum(input_col[2]).over(winSpec),
            )
        )

        # drop local_date_to_Unix
        dataset2 = dataset2.drop("local_date_to_Unix")
        dataset2.na.drop()

        return dataset2
