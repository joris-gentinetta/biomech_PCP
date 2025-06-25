# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Directories
plot_folder <- "/Users/jorisg/Desktop/upper_limb/paper_figures_revision-R"
data_folder <- "/Users/jorisg/Desktop/upper_limb/paper_data-2"

df <- read.csv(file.path(data_folder, 'dataAmount.csv'))
df$"Online Training Time [s]" <- df$X
df$dummy <- "Required Data Amount"
df$"Total MSE" <- df$Value

colors = c('#f89231', '#8323dc')
# Midpoint calculation
midpoint <- max(df$`Online Training Time [s]`) / 2

last_blue_point <- df %>%
  filter(Label == 'Regular') %>%
  arrange(`Online Training Time [s]`) %>%
  slice_tail(n = 1)

last_blue_value <- last_blue_point$`Total MSE`
last_blue_x <- last_blue_point$`Online Training Time [s]`

# Find the closest x-value in the orange line where MSE is approximately equal to last_blue_value
orange_intersection <- df %>%
  filter(Label == 'Double BS, Half SL') %>%
  mutate(difference = abs(`Total MSE` - last_blue_value)) %>%
  arrange(difference) %>%
  slice(2)

intersection_x <- orange_intersection$`Online Training Time [s]`
intersection_y <- orange_intersection$`Total MSE`

combined_plot <- ggplot(df, aes(x = `Online Training Time [s]`, y = `Total MSE`, color = Color, group = Label)) +
  geom_line(data = subset(df, `Online Training Time [s]` <= midpoint ), aes(linetype = "solid")) +
  geom_line(data = subset(df, Label == "Regular"), aes(linetype = "solid")) +
  geom_line(data = subset(df, `Online Training Time [s]` > midpoint -1), aes(linetype = "11")) +
  geom_segment(
    aes(x = midpoint, xend = last_blue_x, y = last_blue_value, yend = last_blue_value),
    linetype = "dotted", color = 'black'
  ) + 
  geom_segment(
    aes(x = intersection_x, xend = intersection_x, y = intersection_y, yend = 0.05),
    linetype = "dotted",
    color = "black"
  ) + 
  facet_grid(
    cols = vars(dummy),
    switch = "y"
  ) +
  scale_color_manual(
    values = colors, 
    labels = unique(df$Label)
  ) +
  scale_linetype_identity() + # Use identity to apply linetype directly
  guides(
    color = guide_legend(title = "Parameters", ncol = 1, title.position = "top", title.hjust = 0.5),
    linetype = "none"
  ) +
  scale_x_continuous(
    breaks = seq(0, 34, by = 34 / 4),  # Specify breaks every 60 units
    labels = c(0, 90, 180, 270, 360)       # Label these breaks as 0 to 6
  ) +  
  theme(
    strip.text.y.left = element_text(angle = 0, face = "bold"),
    strip.placement = "outside",
    plot.title = element_text(face = "bold"),
    legend.position = c(0.98, 0.95), 
    legend.justification = c("right", "top"), 
    legend.direction = "horizontal", 
    legend.box = "horizontal", # Display color and linetype legends in one line
    legend.spacing = unit(0, "lines"), # Remove spacing between legend items
    legend.key = element_blank(),
    legend.title = element_text(size = 12, face = "bold"), 
    legend.background = element_rect(fill = "gray90"), # Sets background for entire legend area
    strip.text = element_text(size = 12, face = "bold"),
    axis.ticks.length = unit(0, "pt"), 
    panel.background = element_rect(fill = "gray98", color = NA), 
    panel.grid.minor = element_blank(), 
    panel.grid.major = element_line(size = 0.3, color = "gray90")
  )


ggsave(
  filename = file.path(plot_folder, "dataAmount.pdf"),
  plot = combined_plot,
  width = 9 * 0.7,
  height = 4 * 0.7,
  dpi = 600,
  device = "pdf"
)
