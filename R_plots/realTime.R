# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Directories
plot_folder <- "/Users/jg/Desktop/upper_limb/paper_figures-4"
data_folder <- "/Users/jg/Desktop/upper_limb/paper_data-2"

df <- read.csv(file.path(data_folder, 'realTime.csv'))
df$MSE <- df$Value
df$"Online Training Time [s]" <- df$X
df$dummy <- "Real-Time Performance"
# colors <- c("#53baa0", "#f89231", "#63a122", "#8323dc", "#3ba7e5", "#d77cdd", "#35618f")
colors = c('#3ba7e5', '#63a122')
df$Label <- factor(df$Label, levels = c("Regular", "Perturbed"))
df$Mode <- factor(df$Mode, levels = c("Offline Stream", "Interpolated Offline Stream", "Real-Time"))


# Create the plot
combined_plot <-ggplot(df, aes(x = factor(Mode), y = Value, fill = Label)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = c("Regular" = "#d77cdd", "Perturbed" = "#35618f")) +
  labs(x = "Training Data", y = "Total MSE") +
  
  facet_grid(
    cols = vars(dummy), 
    switch = "y"
  )  +
  theme(
        strip.placement = "outside",
    legend.title = element_blank(),
    legend.position = c(0.98, 0.25), 
    legend.justification = c("right", "top"), 
    legend.direction = "horizontal", 
    legend.box = "horizontal", # Display color and linetype legends in one line
    legend.spacing = unit(0, "lines"), # Remove spacing between legend items

    legend.key = element_blank(),
    # legend.title = element_text(size = 12, face = "bold"), 
    legend.background = element_rect(fill = "gray90"), # Sets background for entire legend area
    strip.text = element_text(size = 12, face = "bold"),
    axis.ticks.length = unit(0, "pt"), 
    panel.background = element_rect(fill = "gray98", color = NA), 
    panel.grid.minor = element_blank(), 
    panel.grid.major = element_line(size = 0.3)
  ) 





ggsave(
  filename = file.path(plot_folder, "realTime.png"),
  plot = combined_plot,
  width = 9 * 0.7,
  height = 4 * 0.7,
  dpi = 600
)

ggsave(
  filename = file.path(plot_folder, "realTime.eps"),
  plot = combined_plot,
  width = 9 * 0.7,
  height = 4 * 0.7,
  dpi = 600,
  device = "eps"
)
