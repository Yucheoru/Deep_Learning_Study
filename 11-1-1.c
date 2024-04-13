#include <stdio.h>

int main(void)
{
    int arr[5];
    int i;
    int sum = 0;
    int max, min;
    for(i=0; i<5; i++)
    {
        scanf("%d", &arr[i]);
        sum += arr[i];
        if(max < arr[i])
            max=arr[i];
        if(min > arr[i])
            min=arr[i];
    }
    printf("최댓값: %d \n", max);
    printf("최솟값: %d \n", min);
    printf("총 합: %d \n", sum);
    return 0;
}