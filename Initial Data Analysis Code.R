# Data Cleanup and load
data = read.csv(file.choose())
data.mod = data[,-c(2, 4, 6, 8)]

library(ggplot2)

# Boxplot of Diameter
ggplot(data.mod, aes(x = d.D, y = CWS)) + geom_point()
boxplot(data.mod$Diameter, xlab = "Diameter", ylab = "Km", main = "Boxplot of Diameter")

# Summary Statistics of diameter
summary(data.mod$Diameter)

mod = lm(CWS ~ d.D, data = data.mod)

plot(mod, which = 2)

# Histogram and QQ plot for diameter
par(mfrow = c(1,1))
density_plot <- ggplot(data = data.mod, aes(x = Diameter)) +
  geom_histogram(aes(y = ..density..), colour = "black", fill = "white") +
  geom_density(colour = "blue") +
  stat_function(fun = dnorm,
                args = list(mean = mean(data.mod$Diameter), sd = sd(data.mod$Diameter))) +
  labs(x = "Diameter (km)", y = "Density",
       title = "Histogram and density estimates of Diameter")

density_plot

qq_plot  = ggplot(data = data.mod, aes(sample = Diameter)) + stat_qq() + stat_qq_line() +
  labs(x = "Theoretical Normal quantiles", y = "Sample quantiles",
       title = "QQ plot for Diameter")
qq_plot

